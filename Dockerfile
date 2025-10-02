# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
# If Torch install ever fails in your environment, you can use CPU wheels explicitly:
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
#     pip install --no-cache-dir -r requirements.txt

# Optional: pre-download the model to speed up cold starts
RUN python - <<'PY'
from transformers import pipeline
p = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
print('Model ready:', p.model.name_or_path)
PY

# Bring in the app code (including static assets)
COPY . .

# Defaults (can be overridden at runtime)
ENV MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english \
    NEUTRAL_THRESHOLD=0.6 \
    MAX_BATCH=64

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
