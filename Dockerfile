# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# ── Final stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System dependencies (run as root)
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl ca-certificates && \
  rm -rf /var/lib/apt/lists/*

# Create non-root user required by Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user

# Install wheels (still as root is fine)
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Switch to non-root user
USER user
ENV HOME=/home/user \
  PATH=/home/user/.local/bin:$PATH \
  API_BASE_URL="" \
  MODEL_NAME="" \
  HF_TOKEN="" \
  OPENAI_API_KEY="" \
  # Tell sentence-transformers where to store / find models
  SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence_transformers \
  HF_HOME=/home/user/.cache/huggingface \
  TRANSFORMERS_CACHE=/home/user/.cache/huggingface/hub

WORKDIR $HOME/app

# Pre-download the SentenceTransformer model into the image so inference
# never needs to hit the network at runtime.
RUN python -c "\
  import os; \
  os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/home/user/.cache/sentence_transformers'; \
  os.environ['HF_HOME']                    = '/home/user/.cache/huggingface'; \
  os.environ['TRANSFORMERS_CACHE']         = '/home/user/.cache/huggingface/hub'; \
  os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'; \
  from sentence_transformers import SentenceTransformer; \
  SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY --chown=user openenv.yaml my_env.py inference.py pyproject.toml ./
COPY --chown=user server/  ./server/
COPY --chown=user graders/ ./graders/
COPY --chown=user data/    ./data/

# Healthcheck – the / endpoint ALWAYS returns HTTP 200 now, so we just need
# to verify the process is alive.  Use generous timing for slow starts.
HEALTHCHECK --interval=15s --timeout=10s --start-period=45s --retries=5 \
  CMD curl -sf http://localhost:7860/ || exit 1

# Start FastAPI server
CMD ["python", "-m", "server.app"]
