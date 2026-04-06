# Build stage
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Final stage
FROM python:3.11-slim

# Create a non-root user for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    API_BASE_URL="" \
    MODEL_NAME="" \
    HF_TOKEN=""

WORKDIR $HOME/app

# Install dependencies from builder
COPY --chown=user --from=builder /app/wheels /wheels
COPY --chown=user requirements.txt .
RUN pip install --no-cache /wheels/*

# Copy application code
COPY --chown=user openenv.yaml .
COPY --chown=user my_env.py .
COPY --chown=user inference.py .
COPY --chown=user app.py .
COPY --chown=user graders/ ./graders/
COPY --chown=user data/ ./data/

# Healthcheck to verify the app can start
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import my_env; print(1)" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
