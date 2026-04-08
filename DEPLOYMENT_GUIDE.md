# Deploying Content Moderation OpenEnv to Hugging Face Spaces

This project is fully containerized and ready for deployment on Hugging Face Spaces using Docker.

## Step-by-Step Guide

### 1. Create a New Space
- Go to [huggingface.co/new-space](https://huggingface.co/new-space).
- **Space Name**: Use a descriptive name (e.g., `moderix-agent`).
- **Space SDK**: Select **Docker**.
- **Docker Template**: Select **Blank**.

### 2. Hardware Requirements
- **Recommended**: `vcpu=2, memory=8gb` (CPU Basic or Small).
- The project includes `sentence-transformers`, so 8GB RAM ensures smooth semantic grading.

### 3. Sync Repository
- You can upload files manually or connect your GitHub repository `sai21-learn/Moderix` for automatic sync.

### 4. Environment Variables (Critical) 🔐
Go to **Settings > Variables and Secrets** in your Space and add:
- `OPENAI_API_KEY`: Your valid OpenAI API key.
- `MODEL_NAME`: (Optional) Defaults to `gpt-4o-mini`.
- `API_BASE_URL`: (Optional) Defaults to `https://api.openai.com/v1`.

### 5. How it Works
- Hugging Face will detect the `Dockerfile` automatically.
- It builds a multi-stage, non-root image.
- The server (`app.py`) runs on port `7860`.
- The Space will show **Running 🟢** once the health check passes.

## Verification
The Space URL will now respond to:
- `GET /` -> 200 OK (Health Check)
- `POST /reset` -> Resets the environment context.
- `POST /step` -> Processes agent actions.
