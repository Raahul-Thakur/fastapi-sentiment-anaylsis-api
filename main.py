from __future__ import annotations
import os
import asyncio
from typing import List, Literal, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
NEUTRAL_THRESHOLD = float(os.getenv("NEUTRAL_THRESHOLD", "0.6"))  # server default
MAX_BATCH = int(os.getenv("MAX_BATCH", "64"))

# -----------------------------
# Model holder (initialized in lifespan)
# -----------------------------
_sentiment_pipeline = None  # type: ignore

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the HF pipeline on startup; optional cleanup on shutdown."""
    global _sentiment_pipeline

    def _init():
        return pipeline("sentiment-analysis", model=MODEL_NAME)

    _sentiment_pipeline = await asyncio.to_thread(_init)  # startup
    yield
    # Optional: cleanup on shutdown
    # _sentiment_pipeline = None

# -----------------------------
# App init
# -----------------------------
app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "FastAPI microservice using Hugging Face Transformers.\n\n"
        "Model: distilbert-base-uncased-finetuned-sst-2-english (binary).\n"
        "Adds an optional 'neutral' via a threshold on the max score."
    ),
    version="1.1.0",
    lifespan=lifespan,  # <- modern replacement for @app.on_event("startup")
)

# Serve /static/* (CSS, images, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------
# Schemas
# -----------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to analyze")
    neutral_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Override server neutral threshold for this request (0..1)."
    )

class SentimentLabel(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability / band-derived confidence")

class PredictResponse(BaseModel):
    sentiment: SentimentLabel
    model: str = Field(default=MODEL_NAME)
    neutral_threshold: float = Field(..., ge=0.0, le=1.0, description="Threshold actually used for this prediction")

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=MAX_BATCH, description="List of texts")
    neutral_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Override server neutral threshold for this batch (0..1)."
    )

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]

# -----------------------------
# Utils
# -----------------------------
LABEL_MAP = {"POSITIVE": "positive", "NEGATIVE": "negative"}

def postprocess_label(raw_label: str, score: float, threshold: float) -> SentimentLabel:
    """
    Map HF binary output to 3-class with a neutral band.
    If score >= threshold -> positive/negative; else -> neutral.
    Neutral confidence is (1 - score).
    """
    if score >= threshold:
        mapped = LABEL_MAP.get(raw_label.upper())
        if mapped is None:
            raise HTTPException(status_code=500, detail=f"Unknown label from model: {raw_label}")
        return SentimentLabel(label=mapped, confidence=score)
    return SentimentLabel(label="neutral", confidence=1.0 - score)

async def run_pipeline(texts: List[str]):
    if _sentiment_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return await asyncio.to_thread(_sentiment_pipeline, texts)

# -----------------------------
# Meta & Home (Playground)
# -----------------------------
@app.get("/meta", tags=["meta"])
async def meta():
    return {
        "service": "sentiment-api",
        "model": MODEL_NAME,
        "neutral_threshold": NEUTRAL_THRESHOLD,  # server default
    }

@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def home():
    # Minimal, dependency-free UI with a neutral-threshold slider
    return """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentiment Playground</title>
<link rel="stylesheet" href="/static/style.css">
</head><body>
  <h1>Sentiment Playground</h1>
  <p class="sub">Paste one line and click <b>Analyze</b> — or multiple lines (one per line) and click <b>Batch</b>.</p>

  <textarea id="inp" placeholder="I absolutely love this product!&#10;meh&#10;This is the worst."></textarea>

  <div class="controls">
    <button class="primary" id="singleBtn">Analyze</button>
    <button class="secondary" id="batchBtn">Batch</button>
    <div class="slider">
      <label for="thr"><small>Neutral if score &lt; θ</small></label>
      <input id="thr" type="range" min="0" max="1" step="0.01">
      <code id="thrval"></code>
    </div>
    <small>Model: <code id="modelName"></code></small>
  </div>

  <div id="out"></div>

<script>
(async ()=>{
  try{
    const r = await fetch('/meta'); if(r.ok){
      const j = await r.json();
      document.getElementById('modelName').textContent = j.model;
      const thr = document.getElementById('thr'), tv = document.getElementById('thrval');
      thr.value = j.neutral_threshold; tv.textContent = Number(j.neutral_threshold).toFixed(2);
      thr.addEventListener('input', ()=>{ tv.textContent = Number(thr.value).toFixed(2); });
    }
  }catch(e){}
})();

function chip(label, conf){
  const cls = label==='positive'?'pos':(label==='negative'?'neg':'neu');
  return `<span class="chip ${cls}">${label}</span> <small>confidence: ${(conf*100).toFixed(1)}%</small>`;
}

async function post(url, body){
  const r = await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok) throw new Error(await r.text()); return r.json();
}

document.getElementById('singleBtn').onclick = async ()=>{
  const text = document.getElementById('inp').value.trim();
  if(!text) return;
  const el = document.getElementById('out'); el.innerHTML='Processing...';
  const thr = Number(document.getElementById('thr').value);
  try{
    const j = await post('/predict',{text, neutral_threshold: thr});
    el.innerHTML = chip(j.sentiment.label, j.sentiment.confidence);
  }catch(err){ el.innerHTML = '<pre>'+String(err.message||err)+'</pre>'; }
};

document.getElementById('batchBtn').onclick = async ()=>{
  const lines = document.getElementById('inp').value.split('\\n').map(s=>s.trim()).filter(Boolean);
  if(lines.length===0) return;
  const el = document.getElementById('out'); el.innerHTML='Processing...';
  const thr = Number(document.getElementById('thr').value);
  try{
    const j = await post('/predict/batch',{texts:lines, neutral_threshold: thr});
    const rows = j.results.map((res,i)=>`<tr><td>${i+1}</td><td>${lines[i]}</td><td>${chip(res.sentiment.label, res.sentiment.confidence)}</td></tr>`).join('');
    el.innerHTML = `<table><thead><tr><th>#</th><th>text</th><th>prediction</th></tr></thead><tbody>${rows}</tbody></table>`;
  }catch(err){ el.innerHTML = '<pre>'+String(err.message||err)+'</pre>'; }
};
</script>
</body></html>"""

# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/health", tags=["meta"])
async def health():
    return {"ok": _sentiment_pipeline is not None}

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(req: PredictRequest):
    used_threshold = req.neutral_threshold if req.neutral_threshold is not None else NEUTRAL_THRESHOLD
    outputs = await run_pipeline([req.text])  # [{'label': 'POSITIVE', 'score': 0.99}]
    out = outputs[0]
    sentiment = postprocess_label(out["label"], float(out["score"]), used_threshold)
    return PredictResponse(sentiment=sentiment, neutral_threshold=used_threshold)

@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
async def predict_batch(req: BatchPredictRequest):
    if len(req.texts) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"Batch too large. Max is {MAX_BATCH}")
    used_threshold = req.neutral_threshold if req.neutral_threshold is not None else NEUTRAL_THRESHOLD
    outputs = await run_pipeline(req.texts)
    results: List[PredictResponse] = []
    for out in outputs:
        sentiment = postprocess_label(out["label"], float(out["score"]), used_threshold)
        results.append(PredictResponse(sentiment=sentiment, neutral_threshold=used_threshold))
    return BatchPredictResponse(results=results)

# Dev run:
# uvicorn main:app --reload --port 8000
# Or:
if __name__ == "__main__":
  import uvicorn
  uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
