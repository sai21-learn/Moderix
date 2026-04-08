# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .

# Increase timeout for large packages and BUILD ALL WHEELS (including deps)
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels \
    --default-timeout=1000 -r requirements.txt

# ── Final stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl ca-certificates && \
  rm -rf /var/lib/apt/lists/*

# Create non-root user (UID 1000 for HF Spaces)
RUN useradd -m -u 1000 user

# Install wheels from the local folder ONLY (no network hits)
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*

# Switch to non-root user
USER user
ENV HOME=/home/user \
  PATH=/home/user/.local/bin:$PATH \
  SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence_transformers \
  HF_HOME=/home/user/.cache/huggingface \
  TRANSFORMERS_CACHE=/home/user/.cache/huggingface/hub

WORKDIR $HOME/app

# Pre-download SentenceTransformer model into the image
RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY --chown=user openenv.yaml my_env.py inference.py pyproject.toml ./
COPY --chown=user server/  ./server/
COPY --chown=user graders/ ./graders/
COPY --chown=user data/    ./data/

# Healthcheck
HEALTHCHECK --interval=20s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -sf http://localhost:7860/health || exit 1

# Start FastAPI server
CMD ["python", "-m", "server.app"]
