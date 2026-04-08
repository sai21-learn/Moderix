# Build stage
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Final stage
FROM python:3.11-slim

# Install system dependencies (must be root)
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl ca-certificates && \
  rm -rf /var/lib/apt/lists/*

# Create a non-root user for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user

# Install dependencies into system or user space (as root is fine here for wheels)
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Switch to non-root user
USER user
ENV HOME=/home/user \
  PATH=/home/user/.local/bin:$PATH \
  API_BASE_URL="" \
  MODEL_NAME="" \
  HF_TOKEN="" \
  OPENAI_API_KEY=""

WORKDIR $HOME/app

# Pre-download SentenceTransformer model into the user's cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY --chown=user openenv.yaml my_env.py inference.py pyproject.toml uv.lock ./
COPY --chown=user server/ ./server/
COPY --chown=user graders/ ./graders/
COPY --chown=user data/ ./data/

# Healthcheck to verify the app can start
HEALTHCHECK --interval=20s --timeout=30s --start-period=30s --retries=3 \
  CMD (curl -f http://localhost:7860/ || python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')") || exit 1

# Start FastAPI server
CMD ["python", "-m", "server.app"]
