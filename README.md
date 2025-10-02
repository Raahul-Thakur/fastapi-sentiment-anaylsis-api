# Sentiment Analysis API (FastAPI + Transformers)

A tiny, production-ready microservice that classifies text as **positive**, **negative**, or **neutral** with a confidence score.

- ✅ One-page **Playground UI** at `/` (paste text → click **Analyze**)
- ✅ REST endpoints: `/predict`, `/predict/batch`
- ✅ Auto-docs: `/docs` (Swagger), `/redoc`
- ✅ Render-ready via `render.yaml`
- ✅ Neutral **threshold slider** (θ) on the homepage

> **About “neutral”**  
> The default model, `distilbert-base-uncased-finetuned-sst-2-english`, is binary (pos/neg).  
> We add **neutral** by applying a decision **threshold θ** to the model’s max score.  
> You can tweak θ in the UI slider or via the API (`neutral_threshold` parameter).

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Quickstart (Local)](#quickstart-local)
- [Open](#open)
- [Try It](#try-it)
- [API](#api)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)
- [License](#license)

---

## Features
- **FastAPI** app with Pydantic models and async-friendly inference  
- **Hugging Face Transformers** `pipeline("sentiment-analysis")`  
- **Neutral threshold** (θ) control per request (UI slider & API)  
- **Batch** endpoint with max size (configurable)  
- **Health check** at `/health`  
- **Render blueprint** with persistent Hugging Face cache disk  

---

## Project Structure
```bash
├─ main.py              # FastAPI app + Playground UI + endpoints
├─ requirements.txt     # fastapi, uvicorn[standard], transformers, torch
├─ render.yaml          # Render blueprint (build/start, env vars, disk, health)
├─ .gitignore
├─ .dockerignore        # if using Dockerfile
├─ Dockerfile           # optional: container deployment
└─ static/
   └─ style.css         # UI styles
```
---

## Quickstart (Local)

**Python 3.10+ recommended**

~~~bash
# 1) Create & activate a venv
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

# 2) Install deps
pip install --upgrade pip
pip install -r requirements.txt
# If Torch fails on your machine:
# pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt

# 3) Run the API (dev)
uvicorn main:app --reload --port 8000
~~~

---

## Open

- Playground UI: <http://127.0.0.1:8000/>
- Swagger: <http://127.0.0.1:8000/docs>
- ReDoc: <http://127.0.0.1:8000/redoc>
- Health: <http://127.0.0.1:8000/health>

---

## Try It

### Single (UI)
- Paste: `I absolutely love this product!` → **Analyze**  
- Move the **θ** slider to see flips between **neutral** vs **pos/neg**.

### Batch (UI)
- Paste multiple lines (one per line), then **Batch**.

**Handy batch block (paste all lines together):**
```text
I absolutely love this!
This is terrible.
It's fine.
Not bad at all.
Great camera, awful battery.
Great, another crash right before the deadline.
Thanks for the quick fix, it works perfectly now!
Revenue missed estimates and guidance was lowered.
LOL that was actually pretty good 😂
Meh
```

### cURL (CLI)

**Single**
~~~bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Great camera, awful battery.","neutral_threshold":0.6}'
~~~

**Batch**
~~~bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["I love this!","This is terrible.","Meh"],"neutral_threshold":0.7}'
~~~

**Example response**
~~~json
{
  "sentiment": { "label": "positive", "confidence": 0.9987 },
  "model": "distilbert-base-uncased-finetuned-sst-2-english",
  "neutral_threshold": 0.6
}
~~~

---

## API

### `POST /predict`

**Request**
~~~json
{ "text": "I love this!", "neutral_threshold": 0.6 }
~~~
*Note:* `neutral_threshold` is optional; if omitted, the server default is used.

**Response**
~~~json
{
  "sentiment": { "label": "positive|negative|neutral", "confidence": 0.0-1.0 },
  "model": "…",
  "neutral_threshold": 0.6
}
~~~

---

### `POST /predict/batch`

**Request**
~~~json
{ "texts": ["a", "b", "c"], "neutral_threshold": 0.7 }
~~~
*Max batch size* defaults to **64** (configurable via `MAX_BATCH`).

**Response**
~~~json
{
  "results": [
    { "sentiment": { "label": "…", "confidence": 0.0 }, "model": "…", "neutral_threshold": 0.7 },
    { "sentiment": { "label": "…", "confidence": 0.0 }, "model": "…", "neutral_threshold": 0.7 }
  ]
}
~~~

---

### `GET /health`
~~~json
{ "ok": true }
~~~

### `GET /meta`
Returns service metadata used by the UI.

---

## Configuration

Set via environment variables:

| Variable             | Default                                           | Description                                  |
|---------------------|---------------------------------------------------|----------------------------------------------|
| `MODEL_NAME`        | `distilbert-base-uncased-finetuned-sst-2-english` | HF model to load                             |
| `NEUTRAL_THRESHOLD` | `0.6`                                             | Server default θ (0..1)                       |
| `MAX_BATCH`         | `64`                                              | Max items in `/predict/batch`                |
| `HF_HOME`           | *(unset)*                                         | Path to cache HF models (e.g., mounted disk) |

> Prefer a native 3-class model? Try `cardiffnlp/twitter-roberta-base-sentiment-latest` and you can ignore neutral-threshold logic.

**CORS (optional, if calling from a browser app):**
~~~python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)
~~~

---

## Troubleshooting

- **Torch install issues:** use the CPU wheel index (see above).  
- **Cold starts on free tiers:**
  - **Render (Native):** mount a disk and set `HF_HOME` to cache weights.
  - **Docker:** the `Dockerfile` pre-downloads the model to speed first request.
- **Reload vs workers:** `--reload` is for local dev; don’t combine with `--workers`.  
- **Model not loaded yet:** first hit may be slower; check `/health` and logs.

---

## Credits

- Model: [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)  
- Built with [FastAPI](https://fastapi.tiangolo.com/) and [Transformers](https://huggingface.co/docs/transformers)

---

## License

MIT
