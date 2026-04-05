# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY openenv.yaml .
COPY my_env.py .
COPY inference.py .
COPY graders/ ./graders/
COPY data/ ./data/

# Environment configurations (OpenEnv Hackathon required)
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""

CMD ["python", "inference.py"]
