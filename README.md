# CDW server

JSON API for collaborative constitution writing: topic coverage, paragraph
recommendation, and coalition mediation. Routes and payloads are documented
at the top of `app.py`.

Live at `https://talmonlab.bgu.ac.il/cdw/` (e.g. `GET /cdw/health`).

Run locally:

```
pip install -r requirements.txt
waitress-serve --host 127.0.0.1 --port 4201 app:app
```

The first start downloads two models from Hugging Face (over 1 GB).
