# app.py (CDW server)
"""
Unified Flask Server (Topics + Recommend + Mediators)
----------------------------------------------------

This server exposes 3 independent modules (routes). Each request is a single JSON call.

Routes
------
GET  /health
  - Simple status + cached-state info.

POST /topics/compute   (Topic coverage vector)
  Purpose:
    Given a list of paragraph objects, compute a global topic-coverage vector q
    using TF-IDF + NMF (with automatic K selection).

  Input JSON:
    {
      "paragraphs": [
        {"text": "..."},                 # required
        {"text": "...", "pid": "p2"}     # extra fields allowed but ignored
      ]
    }

  Output JSON (success: HTTP 200):
    [q1, q2, ..., qK]   # list[float], sums to 1 (up to floating error)

  Output JSON (error: HTTP 400/500):
    {"ok": false, "error": "<error_code>"}


POST /recommend   (Recommendation for ONE agent)
  Purpose:
    Recommend a single paragraph id (pid) that the agent has not yet voted on,
    using SBERT relevance + topic-coverage gain.

  Input JSON:
    {
      "agent_id": "a1",                          # required (string)

      "paragraphs": {                            # required (dict: pid -> text)
        "p1": "text...",
        "p2": "text..."
      },

      "vote_matrix": {                           # required (dict-of-dicts)
        "a1": {"p1":  1, "p2":  0, "p3": -1},
        "a2": {"p1":  0, "p2":  1, "p3":  0}
      },

      "global_coverage": [0.2, 0.3, 0.5],        # required (list[float], K dims)

      "alpha": 0.5,                              # optional in [0,1] (defaults env ALPHA)
      "lambda": 0.5                              # optional >=0 (defaults env LAMBDA_NEG)
    }

    Notes:
      - vote values:  1 = positive, -1 = negative, 0 = not yet voted.
      - Recommendation is chosen among pids with agent vote == 0.

  Output JSON (success: HTTP 200):
    {"ok": true, "pid": "p7"}                    # pid may be null if no unvoted exists

  Output JSON (error: HTTP 400/500):
    {"ok": false, "error": "<error_code>"}


POST /mediate   (Mediators module: pair coalitions + merge)
  Purpose:
    In ONE call, receive ALL coalition-documents and return, for EACH input id,
    either:
      - a suggested "new document" (list[str]) produced by merging it with a paired coalition, or
      - null, if it was not paired (odd number of non-empty docs) OR if its document is empty.

  Pairing:
    - Only NON-EMPTY documents are eligible for pairing.
    - Each non-empty document is embedded (mean of sentence embeddings, L2-normalized).
    - All pairwise cosine similarities are computed.
    - Greedy matching picks the highest-similarity available pair repeatedly.
    - EXCLUSIONS: Pairs listed in "rejected_pairs" are skipped during matching.

  Merging:
    - For each matched pair, merge their sentences via HDBSCAN clustering over sentence embeddings.
    - Output representative sentences per cluster (centroid-closest), plus optional noise sentences
      according to keep_noise_ratio (deterministic prefix of noise indices).
    - CONTRADICTION FILTERING: If NLI is available, contradictory sentences are filtered using:
      * NLI model identifies semantic contradictions
      * When conflict detected: keep sentence closer to centroid
      * Find non-contradicting replacement closest to rejected sentence
      * Guarantees minimum of 2 sentences in output (when possible)
    - The SAME merged list is returned for BOTH ids in the pair.

  Input JSON (two equivalent formats supported):

    Format A:
    {
      "documents": [
        {"id": "c1", "sentences": ["...", "..."]},
        {"id": "c2", "sentences": ["..."]},
        {"id": "c3", "sentences": []}
      ],

      "rejected_pairs": [["c1", "c4"], ["c2", "c5"]],  # optional: pairs to exclude from matching
      "min_cluster_size": 3,                           # optional (>=2), default env MIN_CLUSTER_SIZE
      "keep_noise_ratio": 0.10                         # optional in [0,1], default env KEEP_NOISE_RATIO
    }

    Format B:
    {
      "docs": {
        "c1": ["...", "..."],
        "c2": ["..."],
        "c3": []
      },

      "rejected_pairs": [["c1", "c4"]],  # optional: pairs to exclude
      "min_cluster_size": 3,             # optional (>=2)
      "keep_noise_ratio": 0.10           # optional in [0,1]
    }

  Client-Side Rejection Tracking:
    - Client tracks ONLY the last rejected pairing per coalition
    - When user rejects merge of (c1, c2): add ["c1", "c2"] to rejected_pairs
    - No need to count consecutive rejections
    - Server automatically avoids all pairs in rejected_pairs list
    - Order within pair doesn't matter: ["c1", "c2"] === ["c2", "c1"]

  Output JSON (success: HTTP 200):
    {
      "ok": true,
      "results": {
        "c1": ["merged sentence 1", "merged sentence 2", ...],   # list[str] OR null
        "c2": ["merged sentence 1", "merged sentence 2", ...],
        "c3": null
      }
    }

    Guarantees:
      - Every input id appears exactly once as a key in results.
      - Empty input documents always map to null.
      - If there is an odd number of non-empty docs, exactly one non-empty id maps to null.
      - Merged results contain minimum 2 sentences (when source documents have >=2 total).
      - No contradictory sentences within same merged result (when NLI available).

  Output JSON (error: HTTP 400/500):
    {"ok": false, "error": "<error_code>"}


HTTP / Port 80
--------------
This process serves plain HTTP (no TLS).

Recommended production setup:
  - Put a reverse proxy in front (nginx / IIS / Caddy) listening on :80 (HTTP),
    forwarding to this Flask server on an internal port (e.g., 8000),
    OR run this process directly on :80.

Important:
  - Do NOT set SSL_CERT_FILE / SSL_KEY_FILE for this HTTP-only server.
  - If you need HTTPS, terminate TLS at the reverse proxy (recommended).

Environment variables:
  HOST=0.0.0.0
  PORT=80
  DEBUG=0

Multilingual (Hebrew + English) support:
  EMBED_MODEL     Sentence embedding model (default: paraphrase-multilingual-MiniLM-L12-v2).
                  Must be a multilingual SBERT model; all-MiniLM-L6-v2 is English-only.
  NLI_MODEL       HuggingFace NLI model for contradiction detection
                  (default: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli).
                  Must be a multilingual NLI model; deberta-v3-base-mnli-fever-anli is English-only.

  TF-IDF preprocessing preserves Hebrew characters (Unicode block U+0590–U+05FF) in addition
  to ASCII Latin tokens, enabling topic modelling over mixed-language or Hebrew-only corpora.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from flask import Flask, jsonify, request
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Optional dependency for /mediate (HDBSCAN-based merge)
try:
    import hdbscan  # type: ignore
except Exception:
    hdbscan = None

# Optional dependency for contradiction detection
try:
    from transformers import pipeline
    NLI_AVAILABLE = True
except Exception:
    NLI_AVAILABLE = False


# -----------------------------
# Config
# -----------------------------
# Multilingual model — supports English, Hebrew, and 50+ other languages.
# Override via EMBED_MODEL env var (must be a multilingual SBERT model).
DEFAULT_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Multilingual NLI model — supports English, Hebrew, and 14 other languages (XNLI set).
# Override via NLI_MODEL env var.
DEFAULT_NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

DEFAULT_ALPHA = float(os.getenv("ALPHA", "0.5"))
DEFAULT_LAMBDA = float(os.getenv("LAMBDA_NEG", "0.5"))
DEFAULT_EPS = float(os.getenv("EPS", "1e-9"))

DEFAULT_RANDOM_STATE = int(os.getenv("RANDOM_STATE", "0"))
DEFAULT_MAX_FEATURES = int(os.getenv("MAX_FEATURES", "5000"))
DEFAULT_MAX_ITER = int(os.getenv("NMF_MAX_ITER", "500"))

K_MIN = int(os.getenv("K_MIN", "3"))
K_MAX = int(os.getenv("K_MAX", "15"))
K_PENALTY = float(os.getenv("K_PENALTY", "0.02"))

DEFAULT_MIN_CLUSTER_SIZE = int(os.getenv("MIN_CLUSTER_SIZE", "3"))
DEFAULT_KEEP_NOISE_RATIO = float(os.getenv("KEEP_NOISE_RATIO", "0.10"))
DEFAULT_MIN_MERGED_SENTENCES = int(os.getenv("MIN_MERGED_SENTENCES", "2"))

MEDIATE_MAX_TOTAL_SENTENCES = int(os.getenv("MEDIATE_MAX_TOTAL_SENTENCES", "500"))
MEDIATE_MAX_CHARS_PER_SENTENCE = int(os.getenv("MEDIATE_MAX_CHARS_PER_SENTENCE", "500"))


# -----------------------------
# Flask app (HTTP only)
# -----------------------------
app = Flask(__name__)

# If you are BEHIND a reverse proxy that sets X-Forwarded-* headers, uncomment this.
# If you run Flask directly (no proxy), keep it commented.
#
# from werkzeug.middleware.proxy_fix import ProxyFix
# app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)


# Browser apps on other sites call this API, so allow cross-origin requests.
# CORS_ORIGIN can narrow it to one site (e.g. https://example.org).
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "*")


@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = CORS_ORIGIN
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


# -----------------------------
# Shared preprocessing / tfidf
# -----------------------------
# Hebrew Unicode block: U+0590–U+05FF (includes vowel marks, punctuation, letters).
# We keep only the Hebrew letter range U+05D0–U+05EA for token content.
_HEBREW_WORD_RE = re.compile(r"[\u05d0-\u05ea]+")
_LATIN_WORD_RE  = re.compile(r"[a-z]+")

# Minimal Hebrew function-word stoplist (common prepositions, conjunctions, articles
# that appear as prefixes or standalone tokens after basic splitting).
_HEBREW_STOP_WORDS: frozenset = frozenset({
    "של", "את", "עם", "על", "אל", "כי", "לא", "הוא", "היא", "הם", "הן",
    "אני", "אנו", "אנחנו", "אתה", "את", "אתם", "אתן", "זה", "זו", "זאת",
    "יש", "אין", "כן", "לו", "לה", "להם", "להן", "לי", "לנו", "לכם",
    "מה", "מי", "כל", "עוד", "גם", "רק", "אם", "או", "אבל", "כך", "כן",
    "שם", "פה", "כאן", "שהוא", "שהיא", "שהם", "שיש", "שאין",
})


def preprocess_text(text: str) -> str:
    """
    Normalise text for TF-IDF.  Preserves:
      - ASCII Latin tokens (lowercased, length > 2, not in English stoplist)
      - Hebrew tokens (U+05D0–U+05EA, length > 1, not in Hebrew stoplist)
    All other characters (digits, punctuation, Arabic, etc.) are discarded.
    Mixed Hebrew/English documents are handled correctly.
    """
    text = str(text).lower()

    latin_tokens = [
        t for t in _LATIN_WORD_RE.findall(text)
        if len(t) > 2 and t not in ENGLISH_STOP_WORDS
    ]

    # Hebrew letters are not lowercased (no case distinction in Hebrew).
    # Re-extract from original (pre-lower) text to avoid mojibake on Hebrew.
    hebrew_tokens = [
        t for t in _HEBREW_WORD_RE.findall(str(text))  # original case irrelevant
        if len(t) > 1 and t not in _HEBREW_STOP_WORDS
    ]

    return " ".join(latin_tokens + hebrew_tokens)


def build_tfidf(texts_raw: List[str]):
    cleaned = [preprocess_text(t) for t in texts_raw]
    n = len(cleaned)

    # Fallback: if fewer than half the texts produced any tokens (e.g. very short inputs),
    # re-run preprocessing without stopword filtering as a lenient pass.
    if sum(1 for c in cleaned if c.strip()) < max(1, n // 2):
        cleaned = [
            " ".join(
                _LATIN_WORD_RE.findall(str(t).lower()) +
                _HEBREW_WORD_RE.findall(str(t))
            )
            for t in texts_raw
        ]

    vectorizer = TfidfVectorizer(
        max_features=DEFAULT_MAX_FEATURES,
        sublinear_tf=True,
        stop_words=None,   # stopword removal already done in preprocess_text
        analyzer="word",
        token_pattern=r"(?u)\b[\w\u05d0-\u05ea]+\b",  # matches Latin words AND Hebrew tokens
    )
    return vectorizer.fit_transform(cleaned)


# ============================================================
# 1) /topics/compute  (one-shot global coverage q)
# ============================================================
def _validate_paragraphs_list(items: Any) -> Tuple[bool, str]:
    if not isinstance(items, list) or not items:
        return False, "paragraphs_must_be_nonempty_list"
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            return False, f"paragraph_{i}_must_be_object"
        if "text" not in it or not str(it["text"]).strip():
            return False, f"paragraph_{i}_missing_text"
    return True, ""


def _fit_nmf(tfidf, k: int) -> Tuple[np.ndarray, float]:
    nmf = NMF(
        n_components=k,
        init="nndsvd",
        random_state=DEFAULT_RANDOM_STATE,
        max_iter=DEFAULT_MAX_ITER,
    )
    W = nmf.fit_transform(tfidf)
    return W, float(nmf.reconstruction_err_)


def _choose_k(tfidf, n_docs: int) -> int:
    if n_docs <= 1:
        return 1

    k_min = max(1, K_MIN)
    k_max = min(max(k_min, K_MAX), n_docs)

    _, err1 = _fit_nmf(tfidf, k=1)
    denom = max(err1, 1e-12)

    best_k = 1
    best_score = 0.0

    for k in range(k_min, k_max + 1):
        if k == 1:
            continue
        _, errk = _fit_nmf(tfidf, k=k)
        improvement = (err1 - errk) / denom
        score = improvement - K_PENALTY * (k - 1)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def compute_global_coverage(paragraphs: List[dict]) -> List[float]:
    texts = [p["text"] for p in paragraphs]
    n = len(texts)

    if n == 1:
        return [1.0]

    tfidf = build_tfidf(texts)
    k = _choose_k(tfidf, n)

    if k == 1:
        return [1.0]

    W, _ = _fit_nmf(tfidf, k=k)

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_norm = W / row_sums

    q = W_norm.mean(axis=0)
    q = q / (float(q.sum()) if float(q.sum()) > 0 else 1.0)

    return q.tolist()


@app.post("/topics/compute")
def topics_compute():
    data = request.get_json(force=True, silent=False) or {}
    paragraphs = data.get("paragraphs")

    ok, err = _validate_paragraphs_list(paragraphs)
    if not ok:
        return jsonify({"ok": False, "error": err}), 400

    try:
        coverage = compute_global_coverage(paragraphs)
    except Exception:
        return jsonify({"ok": False, "error": "topic_model_failed"}), 500

    return jsonify(coverage), 200


# ============================================================
# 2) /recommend  (SBERT + TFIDF/NMF fixed-K topic vecs)
# ============================================================
@dataclass
class Paragraph:
    pid: str
    text: str
    emb: Optional[np.ndarray] = None
    topic_vec: Optional[np.ndarray] = None


class RecommenderState:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.paragraphs: Dict[str, Paragraph] = {}
        self.global_coverage: Optional[np.ndarray] = None

        self._paragraphs_hash: Optional[int] = None
        self._topic_k: Optional[int] = None
        self.updated_at: float = time.time()
        
        # Initialize NLI classifier for contradiction detection.
        # Uses a multilingual model (mDeBERTa-v3-base-mnli-xnli by default) that
        # supports English, Hebrew, and 14 other languages in the XNLI set.
        self.nli_classifier = None
        if NLI_AVAILABLE:
            try:
                self.nli_classifier = pipeline(
                    "text-classification",
                    model=DEFAULT_NLI_MODEL,
                    device=-1  # Use CPU; set device=0 for GPU
                )
            except Exception:
                pass  # Silently fail if model cannot be loaded

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + DEFAULT_EPS
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _jsd(a: np.ndarray, b: np.ndarray) -> float:
        return float(jensenshannon(a, b, base=2))

    @staticmethod
    def _clip01(x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)

    @staticmethod
    def validate_recommend_payload(data: Dict[str, Any]) -> Tuple[bool, str]:
        if not isinstance(data, dict):
            return False, "body_must_be_object"
        if "agent_id" not in data or not isinstance(data["agent_id"], str) or not data["agent_id"].strip():
            return False, "agent_id_required"
        if "paragraphs" not in data or not isinstance(data["paragraphs"], dict) or not data["paragraphs"]:
            return False, "paragraphs_must_be_nonempty_dict"
        if "vote_matrix" not in data or not isinstance(data["vote_matrix"], dict) or not data["vote_matrix"]:
            return False, "vote_matrix_must_be_nonempty_dict"
        if "global_coverage" not in data or not isinstance(data["global_coverage"], list) or not data["global_coverage"]:
            return False, "global_coverage_must_be_nonempty_list"
        return True, ""

    def _upsert_paragraphs_from_dict(self, paragraphs_dict: Dict[str, str]) -> None:
        changed = False

        current_pids = set(self.paragraphs.keys())
        incoming_pids = set(str(pid) for pid in paragraphs_dict.keys())
        removed = current_pids - incoming_pids
        if removed:
            for pid in removed:
                del self.paragraphs[pid]
            changed = True

        for pid, text in paragraphs_dict.items():
            pid = str(pid)
            text = str(text)
            if pid not in self.paragraphs:
                self.paragraphs[pid] = Paragraph(pid=pid, text=text)
                changed = True
            else:
                if self.paragraphs[pid].text != text:
                    self.paragraphs[pid].text = text
                    self.paragraphs[pid].emb = None
                    self.paragraphs[pid].topic_vec = None
                    changed = True

        if changed:
            self._paragraphs_hash = None
            self.updated_at = time.time()

    def _compute_paragraphs_hash(self) -> int:
        items = tuple(sorted((pid, self.paragraphs[pid].text) for pid in self.paragraphs.keys()))
        return hash(items)

    def _ensure_embeddings(self) -> None:
        missing = [p for p in self.paragraphs.values() if p.emb is None]
        if not missing:
            return
        texts = [p.text for p in missing]
        embs = self.model.encode(texts, normalize_embeddings=False)
        for para, emb in zip(missing, embs):
            para.emb = np.asarray(emb, dtype=np.float32)

    def _ensure_topic_vectors_fixed_k(self, K: int) -> None:
        current_hash = self._compute_paragraphs_hash()
        need_refit = (
            self._paragraphs_hash != current_hash
            or self._topic_k != K
            or any(p.topic_vec is None for p in self.paragraphs.values())
        )
        if not need_refit:
            return

        n = len(self.paragraphs)
        if n <= 0:
            return

        if n == 1:
            only_pid = next(iter(self.paragraphs.keys()))
            vec = np.zeros(K, dtype=np.float64)
            vec[0] = 1.0
            self.paragraphs[only_pid].topic_vec = vec
            self._paragraphs_hash = current_hash
            self._topic_k = K
            return

        texts_raw = [self.paragraphs[pid].text for pid in sorted(self.paragraphs.keys())]
        tfidf = build_tfidf(texts_raw)

        n_docs, n_feat = tfidf.shape
        K_fit = int(K)
        if K_fit > n_docs:
            K_fit = n_docs
        if K_fit > n_feat:
            K_fit = max(1, n_feat)

        if K_fit <= 1:
            for pid in self.paragraphs.keys():
                vec = np.zeros(K, dtype=np.float64)
                vec[0] = 1.0
                self.paragraphs[pid].topic_vec = vec
            self._paragraphs_hash = current_hash
            self._topic_k = K
            return

        nmf = NMF(
            n_components=K_fit,
            random_state=DEFAULT_RANDOM_STATE,
            init="nndsvd",
            max_iter=DEFAULT_MAX_ITER,
        )
        W = nmf.fit_transform(tfidf)

        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W_norm = W / row_sums

        for idx, pid in enumerate(sorted(self.paragraphs.keys())):
            base = np.asarray(W_norm[idx], dtype=np.float64)
            if K_fit < K:
                padded = np.zeros(K, dtype=np.float64)
                padded[:K_fit] = base
                s = float(padded.sum())
                padded = padded / (s if s > 0 else 1.0)
                self.paragraphs[pid].topic_vec = padded
            else:
                s = float(base.sum())
                base = base / (s if s > 0 else 1.0)
                self.paragraphs[pid].topic_vec = base

        self._paragraphs_hash = current_hash
        self._topic_k = K

    def _extract_agent_votes_row(self, vote_matrix: Dict[str, Any], agent_id: str) -> Dict[str, int]:
        row = vote_matrix.get(agent_id, None)
        if row is None or not isinstance(row, dict):
            return {}
        out: Dict[str, int] = {}
        for pid, v in row.items():
            try:
                out[str(pid)] = int(v)
            except Exception:
                out[str(pid)] = 0
        return out

    def _agent_profile(
        self,
        agent_votes: Dict[str, int],
        K: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, int]:
        pos_embs: List[np.ndarray] = []
        neg_embs: List[np.ndarray] = []
        pos_topics: List[np.ndarray] = []

        for pid, v in agent_votes.items():
            pid = str(pid)
            if pid not in self.paragraphs:
                continue
            if v == 1:
                if self.paragraphs[pid].emb is not None:
                    pos_embs.append(self.paragraphs[pid].emb)
                if self.paragraphs[pid].topic_vec is not None:
                    pos_topics.append(self.paragraphs[pid].topic_vec)
            elif v == -1:
                if self.paragraphs[pid].emb is not None:
                    neg_embs.append(self.paragraphs[pid].emb)

        mu_pos = np.mean(np.stack(pos_embs, axis=0), axis=0) if pos_embs else None
        mu_neg = np.mean(np.stack(neg_embs, axis=0), axis=0) if neg_embs else None

        if pos_topics:
            r = np.mean(np.stack(pos_topics, axis=0), axis=0)
            r = r / (float(r.sum()) if float(r.sum()) > 0 else 1.0)
            return mu_pos, mu_neg, r, len(pos_topics)

        r0 = np.ones(K, dtype=np.float64) / float(K)
        return mu_pos, mu_neg, r0, 0

    def _relevance01(self, pid: str, mu_pos: Optional[np.ndarray], mu_neg: Optional[np.ndarray], lam: float) -> float:
        if mu_pos is None:
            raw = 0.0
        else:
            raw = self._cos(self.paragraphs[pid].emb, mu_pos)
            if mu_neg is not None:
                raw -= lam * self._cos(self.paragraphs[pid].emb, mu_neg)
        return self._clip01((raw + 1.0) / 2.0)

    def _coverage_gain01(self, pid: str, q: np.ndarray, r: np.ndarray, n_pos: int) -> float:
        t = self.paragraphs[pid].topic_vec
        if t is None:
            return 0.5

        d_before = self._jsd(q, r)

        if n_pos == 0:
            r_new = t
        else:
            r_new = (n_pos * r + t) / float(n_pos + 1)
            r_new = r_new / (float(r_new.sum()) if float(r_new.sum()) > 0 else 1.0)

        d_after = self._jsd(q, r_new)
        gain = d_before - d_after
        return self._clip01((gain + 1.0) / 2.0)

    def recommend(
        self,
        agent_id: str,
        vote_matrix: Dict[str, Any],
        paragraphs_dict: Dict[str, str],
        global_coverage_list: List[float],
        alpha: float,
        lam: float,
    ) -> Dict[str, Any]:
        self._upsert_paragraphs_from_dict(paragraphs_dict)

        if not self.paragraphs:
            return {"ok": False, "error": "no_paragraphs_provided"}

        q = np.asarray(global_coverage_list, dtype=np.float64)
        if q.ndim != 1 or q.shape[0] < 1:
            return {"ok": False, "error": "global_coverage_invalid_shape"}
        if np.any(q < 0):
            return {"ok": False, "error": "global_coverage_negative_entries"}

        s = float(q.sum())
        if s <= 0:
            return {"ok": False, "error": "global_coverage_sum_must_be_positive"}
        q = q / s
        K = int(q.shape[0])
        self.global_coverage = q

        self._ensure_embeddings()
        self._ensure_topic_vectors_fixed_k(K)

        agent_votes = self._extract_agent_votes_row(vote_matrix, agent_id)
        if not agent_votes:
            return {"ok": False, "error": "agent_id_not_found_in_vote_matrix"}

        mu_pos, mu_neg, r, n_pos = self._agent_profile(agent_votes, K=K)

        best_pid: Optional[str] = None
        best_score: float = -1e18

        for pid in self.paragraphs.keys():
            v = agent_votes.get(pid, 0)
            try:
                v = int(v)
            except Exception:
                v = 0
            if v != 0:
                continue

            rel01 = self._relevance01(pid, mu_pos, mu_neg, lam=lam)
            cov01 = self._coverage_gain01(pid, q=q, r=r, n_pos=n_pos)
            score = float(alpha * rel01 + (1.0 - alpha) * cov01)

            if score > best_score:
                best_score = score
                best_pid = pid

        return {"ok": True, "pid": best_pid}


STATE = RecommenderState(DEFAULT_MODEL_NAME)


@app.post("/recommend")
def recommend_route():
    data = request.get_json(force=True, silent=False) or {}

    ok, err = STATE.validate_recommend_payload(data)
    if not ok:
        return jsonify({"ok": False, "error": err}), 400

    agent_id = data["agent_id"]
    vote_matrix = data["vote_matrix"]
    paragraphs_dict = data["paragraphs"]
    global_coverage = data["global_coverage"]

    try:
        alpha = float(data.get("alpha", DEFAULT_ALPHA))
        lam = float(data.get("lambda", DEFAULT_LAMBDA))
        alpha = min(1.0, max(0.0, alpha))
        lam = max(0.0, lam)
    except Exception:
        return jsonify({"ok": False, "error": "invalid_alpha_or_lambda"}), 400

    try:
        res = STATE.recommend(
            agent_id=agent_id,
            vote_matrix=vote_matrix,
            paragraphs_dict=paragraphs_dict,
            global_coverage_list=global_coverage,
            alpha=alpha,
            lam=lam,
        )
    except Exception:
        return jsonify({"ok": False, "error": "server_failed"}), 500

    code = 200 if res.get("ok") else 400
    return jsonify(res), code


# ============================================================
# 3) /mediate  (pair coalitions + merge via HDBSCAN) — merge is INTERNAL here
# ============================================================
def _clean_sentence(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _dedupe_sentences_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        x = _clean_sentence(x)
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _parse_mediate_payload(data: Dict[str, Any]) -> Tuple[List[str], List[List[str]], int, float, Set[frozenset]]:
    if not isinstance(data, dict):
        raise ValueError("body_must_be_object")

    min_cluster_size = int(data.get("min_cluster_size", DEFAULT_MIN_CLUSTER_SIZE))
    keep_noise_ratio = float(data.get("keep_noise_ratio", DEFAULT_KEEP_NOISE_RATIO))
    min_cluster_size = max(2, min_cluster_size)
    keep_noise_ratio = min(1.0, max(0.0, keep_noise_ratio))

    ids: List[str] = []
    docs: List[List[str]] = []

    if "documents" in data:
        documents = data.get("documents")
        if not isinstance(documents, list) or len(documents) < 1:
            raise ValueError("documents_must_be_nonempty_list")

        for i, item in enumerate(documents):
            if not isinstance(item, dict):
                raise ValueError(f"documents_{i}_must_be_object")
            cid = item.get("id", None)
            sents = item.get("sentences", None)
            if not isinstance(cid, str) or not cid.strip():
                raise ValueError(f"documents_{i}_missing_id")
            if not isinstance(sents, list) or not all(isinstance(x, (str, int, float)) for x in sents):
                raise ValueError(f"documents_{i}_sentences_must_be_list")
            ids.append(cid.strip())
            docs.append([str(x) for x in sents])

    elif "docs" in data:
        dd = data.get("docs")
        if not isinstance(dd, dict) or len(dd) < 1:
            raise ValueError("docs_must_be_nonempty_object")
        for cid, sents in dd.items():
            if not isinstance(cid, str) or not cid.strip():
                raise ValueError("docs_keys_must_be_strings")
            if not isinstance(sents, list) or not all(isinstance(x, (str, int, float)) for x in sents):
                raise ValueError(f"docs_{cid}_must_be_list")
            ids.append(cid.strip())
            docs.append([str(x) for x in sents])

    else:
        raise ValueError("missing_documents_or_docs")

    if len(ids) < 1:
        raise ValueError("no_documents")
    if len(set(ids)) != len(ids):
        raise ValueError("document_ids_must_be_unique")

    total_sents = 0
    cleaned_docs: List[List[str]] = []
    for d in docs:
        d2 = _dedupe_sentences_keep_order(d)
        for s in d2:
            if len(s) > MEDIATE_MAX_CHARS_PER_SENTENCE:
                raise ValueError("sentence_too_long")
        cleaned_docs.append(d2)
        total_sents += len(d2)

    if total_sents > MEDIATE_MAX_TOTAL_SENTENCES:
        raise ValueError("too_many_total_sentences")
    
    # Parse rejected_pairs
    rejected_pairs_raw = data.get("rejected_pairs", [])
    rejected_pairs: Set[frozenset] = set()
    
    if not isinstance(rejected_pairs_raw, list):
        raise ValueError("rejected_pairs_must_be_list")
    
    for pair in rejected_pairs_raw:
        if not isinstance(pair, list) or len(pair) != 2:
            continue  # Skip malformed pairs silently
        id1, id2 = str(pair[0]).strip(), str(pair[1]).strip()
        if id1 and id2:
            rejected_pairs.add(frozenset([id1, id2]))

    return ids, cleaned_docs, min_cluster_size, keep_noise_ratio, rejected_pairs


def _doc_embedding_mean(model: SentenceTransformer, doc: List[str]) -> np.ndarray:
    dim = model.get_sentence_embedding_dimension()
    if not doc:
        return np.zeros((dim,), dtype=np.float32)

    X = model.encode(doc, normalize_embeddings=False)
    X = np.asarray(X, dtype=np.float32)
    v = X.mean(axis=0)

    n = float(np.linalg.norm(v))
    if n <= 0:
        return np.zeros((dim,), dtype=np.float32)
    return (v / n).astype(np.float32)


def _all_pairwise_sims(V: np.ndarray) -> List[Tuple[int, int, float]]:
    D = V.shape[0]
    S = V @ V.T
    pairs: List[Tuple[int, int, float]] = []
    for i in range(D):
        for j in range(i + 1, D):
            pairs.append((i, j, float(S[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def _greedy_pairing(
    pairs_sorted: List[Tuple[int, int, float]], 
    n_docs: int,
    id_list: List[str],
    rejected_pairs: Set[frozenset]
) -> Tuple[List[Tuple[int, int, float]], List[int]]:
    """
    Greedy pairing with exclusion of previously rejected combinations.
    
    Args:
        pairs_sorted: All pairs sorted by similarity (descending)
        n_docs: Total number of documents
        id_list: Document IDs in order (for checking rejection history)
        rejected_pairs: Set of frozenset pairs to exclude from matching
    
    Returns:
        matched: List of (i, j, similarity) tuples for matched pairs
        leftovers: List of unmatched document indices
    """
    used = set()
    matched: List[Tuple[int, int, float]] = []
    
    for i, j, sim in pairs_sorted:
        if i in used or j in used:
            continue
        
        # Check if this pairing was previously rejected
        pair_ids = frozenset([id_list[i], id_list[j]])
        if pair_ids in rejected_pairs:
            continue  # Skip this pairing
        
        used.add(i)
        used.add(j)
        matched.append((i, j, sim))
        
        if len(used) >= n_docs - 1:
            break
    
    leftovers = [k for k in range(n_docs) if k not in used]
    return matched, leftovers


def _pick_representatives(
    sentences: List[str],
    labels: np.ndarray,
    embs: np.ndarray,
    keep_noise_ratio: float
) -> Tuple[List[str], int, int]:
    merged: List[str] = []
    noise_kept = 0

    clusters = sorted({int(x) for x in labels.tolist() if int(x) >= 0})
    for c in clusters:
        idxs = np.where(labels == c)[0]
        centroid = embs[idxs].mean(axis=0, keepdims=True)
        dists = np.linalg.norm(embs[idxs] - centroid, axis=1)
        rep_i = idxs[int(np.argmin(dists))]
        merged.append(sentences[int(rep_i)])

    noise_idxs = np.where(labels == -1)[0]
    if noise_idxs.size > 0 and keep_noise_ratio > 0:
        k_keep = int(round(float(noise_idxs.size) * float(keep_noise_ratio)))
        k_keep = max(0, min(k_keep, int(noise_idxs.size)))
        if k_keep > 0:
            for i in noise_idxs[:k_keep]:
                merged.append(sentences[int(i)])
            noise_kept = k_keep

    return merged, len(clusters), noise_kept


def _filter_contradictions_smart(
    sentences: List[str],
    embs: np.ndarray,
    nli_classifier,
    min_sentences: int = 2,
    threshold: float = 0.75
) -> List[str]:
    """
    Filter contradictory sentences with replacement strategy.
    
    Strategy:
      1. Build contradiction graph using NLI
      2. Select sentences greedily by distance to centroid
      3. When rejecting a sentence due to contradiction:
         - Find a replacement: closest non-contradicting sentence
      4. Ensure minimum number of sentences
    
    Args:
        sentences: Candidate sentences
        embs: Sentence embeddings (same order as sentences)
        nli_classifier: HuggingFace NLI pipeline
        min_sentences: Minimum sentences to retain (default 2)
        threshold: Contradiction confidence threshold (default 0.75)
    
    Returns:
        Filtered list maintaining diversity and consistency
    """
    if nli_classifier is None or len(sentences) <= min_sentences:
        return sentences
    
    n = len(sentences)
    
    # Compute centroid for distance-based selection
    centroid = embs.mean(axis=0, keepdims=True)
    dists_to_centroid = np.linalg.norm(embs - centroid, axis=1)
    
    # Build contradiction matrix
    contradicts = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                result = nli_classifier(f"{sentences[i]} [SEP] {sentences[j]}")[0]
                if result['label'].lower() == 'contradiction' and result['score'] > threshold:
                    contradicts[i, j] = True
                    contradicts[j, i] = True
            except Exception:
                pass  # Skip on NLI error
    
    # Greedy selection: start with closest to centroid
    order = np.argsort(dists_to_centroid)
    selected = []
    selected_indices = set()
    rejected_indices = set()
    
    for idx in order:
        if idx in rejected_indices:
            continue
        
        # Check if this contradicts any already selected
        is_contradicting = False
        for sel_idx in selected_indices:
            if contradicts[idx, sel_idx]:
                is_contradicting = True
                break
        
        if not is_contradicting:
            selected.append(sentences[idx])
            selected_indices.add(idx)
        else:
            rejected_indices.add(idx)
            
            # Replacement strategy: find closest non-contradicting sentence
            if len(selected) < min_sentences:
                # Find a replacement
                replacement_idx = None
                min_dist = float('inf')
                
                for candidate_idx in range(n):
                    if candidate_idx in selected_indices or candidate_idx in rejected_indices:
                        continue
                    
                    # Check if candidate contradicts any selected
                    candidate_contradicts = False
                    for sel_idx in selected_indices:
                        if contradicts[candidate_idx, sel_idx]:
                            candidate_contradicts = True
                            break
                    
                    if not candidate_contradicts:
                        # Distance to the sentence we would have selected
                        dist = np.linalg.norm(embs[candidate_idx] - embs[idx])
                        if dist < min_dist:
                            min_dist = dist
                            replacement_idx = candidate_idx
                
                if replacement_idx is not None:
                    selected.append(sentences[replacement_idx])
                    selected_indices.add(replacement_idx)
                    rejected_indices.add(replacement_idx)
    
    # Ensure minimum sentences by adding closest non-contradicting ones
    while len(selected) < min_sentences and len(selected_indices) < n:
        best_idx = None
        min_avg_dist = float('inf')
        
        for candidate_idx in range(n):
            if candidate_idx in selected_indices or candidate_idx in rejected_indices:
                continue
            
            # Check contradiction with all selected
            candidate_contradicts = False
            for sel_idx in selected_indices:
                if contradicts[candidate_idx, sel_idx]:
                    candidate_contradicts = True
                    break
            
            if not candidate_contradicts:
                # Compute average distance to selected sentences
                if len(selected_indices) > 0:
                    avg_dist = np.mean([np.linalg.norm(embs[candidate_idx] - embs[sel_idx]) 
                                       for sel_idx in selected_indices])
                else:
                    avg_dist = dists_to_centroid[candidate_idx]
                
                if avg_dist < min_avg_dist:
                    min_avg_dist = avg_dist
                    best_idx = candidate_idx
        
        if best_idx is not None:
            selected.append(sentences[best_idx])
            selected_indices.add(best_idx)
        else:
            # No non-contradicting sentences left, break
            break
    
    return selected


def _merge_two_docs_internal(
    doc_a: List[str],
    doc_b: List[str],
    min_cluster_size: int,
    keep_noise_ratio: float,
    use_nli: bool = True,
    min_merged: int = DEFAULT_MIN_MERGED_SENTENCES
) -> Dict[str, Any]:
    if hdbscan is None:
        raise RuntimeError("hdbscan_not_installed")

    sentences = [s for s in (doc_a + doc_b) if str(s).strip()]
    if not sentences:
        return {"merged": [], "clusters": 0, "noise_kept": 0}

    embs = STATE.model.encode(sentences, normalize_embeddings=False)
    embs = np.asarray(embs, dtype=np.float32)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, int(min_cluster_size)),
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embs)

    merged, n_clusters, noise_kept = _pick_representatives(
        sentences=sentences,
        labels=labels,
        embs=embs,
        keep_noise_ratio=float(keep_noise_ratio),
    )

    merged = _dedupe_sentences_keep_order([str(x) for x in merged])
    
    # Apply contradiction filtering if requested and available
    if use_nli and STATE.nli_classifier is not None and len(merged) > 1:
        merged = _filter_contradictions_smart(
            sentences=merged,
            embs=STATE.model.encode(merged, normalize_embeddings=False),
            nli_classifier=STATE.nli_classifier,
            min_sentences=min_merged
        )
    
    return {"merged": merged, "clusters": int(n_clusters), "noise_kept": int(noise_kept)}


@app.post("/mediate")
def mediate_route():
    if hdbscan is None:
        return jsonify({"ok": False, "error": "hdbscan_not_installed"}), 500

    data = request.get_json(force=True, silent=False) or {}

    try:
        ids, docs, min_cluster_size, keep_noise_ratio, rejected_pairs = _parse_mediate_payload(data)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    D = len(ids)
    results: Dict[str, Any] = {cid: None for cid in ids}

    non_empty = [i for i, d in enumerate(docs) if len(d) > 0]
    if len(non_empty) < 2:
        return jsonify({"ok": True, "results": results}), 200

    try:
        vecs = np.stack([_doc_embedding_mean(STATE.model, d) for d in docs], axis=0)
    except Exception:
        return jsonify({"ok": False, "error": "embedding_failed"}), 500

    try:
        pairs_sorted = _all_pairwise_sims(vecs)
        matched_pairs, leftovers = _greedy_pairing(pairs_sorted, D, ids, rejected_pairs)
    except Exception:
        return jsonify({"ok": False, "error": "pairing_failed"}), 500

    try:
        for i, j, _sim in matched_pairs:
            merged_res = _merge_two_docs_internal(
                docs[i], docs[j],
                min_cluster_size=min_cluster_size,
                keep_noise_ratio=keep_noise_ratio,
                use_nli=True,  # Enable NLI contradiction filtering
                min_merged=DEFAULT_MIN_MERGED_SENTENCES
            )
            suggested = merged_res["merged"]
            results[ids[i]] = suggested
            results[ids[j]] = suggested

        for k in leftovers:
            results[ids[k]] = None
    except Exception:
        return jsonify({"ok": False, "error": "merge_failed"}), 500

    return jsonify({"ok": True, "results": results}), 200


# ============================================================
# Health
# ============================================================
@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "embedding_model": DEFAULT_MODEL_NAME,
            "recommend": {
                "paragraphs_cached": len(STATE.paragraphs),
                "topic_k_cached": STATE._topic_k,
                "topics_ready": STATE._topic_k is not None and all(
                    p.topic_vec is not None for p in STATE.paragraphs.values()
                ),
                "global_coverage_ready": STATE.global_coverage is not None,
                "updated_at": STATE.updated_at,
            },
            "topics": {
                "model": "tfidf+nmf",
                "k_min": K_MIN,
                "k_max": K_MAX,
            },
            "mediate": {
                "hdbscan_available": hdbscan is not None,
                "nli_available": NLI_AVAILABLE and STATE.nli_classifier is not None,
                "nli_model": DEFAULT_NLI_MODEL,
                "default_min_cluster_size": DEFAULT_MIN_CLUSTER_SIZE,
                "default_keep_noise_ratio": DEFAULT_KEEP_NOISE_RATIO,
                "default_min_merged_sentences": DEFAULT_MIN_MERGED_SENTENCES,
            },
        }
    )


# ============================================================
# Run (HTTP only)
# ============================================================
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "80"))
    debug = bool(int(os.getenv("DEBUG", "0")))

    # HTTP ONLY: no TLS/SSL in this process.
    app.run(host=host, port=port, debug=debug, ssl_context=None)