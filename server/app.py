import asyncio
import sys
from fastapi import FastAPI, HTTPException
import my_env
from pydantic import BaseModel

app = FastAPI(title="Content Moderation OpenEnv API")

# Global environment instance
environment: my_env.ContentModerationEnv = None
_init_error: str = None

@app.on_event("startup")
async def startup_event():
    global environment, _init_error
    try:
        # Initialize the Content Moderation RL environment
        environment = await my_env.ContentModerationEnv.from_env()
        print("[INFO] Environment initialized successfully", file=sys.stderr, flush=True)
    except Exception as e:
        _init_error = str(e)
        print(f"[ERROR] Environment initialization failed: {e}", file=sys.stderr, flush=True)
        # Create a minimal fallback environment so the server stays alive
        try:
            environment = my_env.ContentModerationEnv()
        except Exception:
            pass

@app.get("/")
async def health_check():
    """Hugging Face Spaces automated ping / healthcheck endpoint - always returns 200."""
    # Always return 200 so Docker/HF healthchecks never fail due to initialisation lag.
    status = "ready" if environment is not None else "initializing"
    return {"status": status, "error": _init_error}

@app.get("/health")
async def health_detailed():
    """Detailed health endpoint."""
    return {"status": "healthy", "env_ready": environment is not None}

class ResetRequest(BaseModel):
    task_id: str = None

@app.post("/reset")
async def reset_env(request: ResetRequest = None):
    """Reset the environment state for a new episode"""
    if environment is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    try:
        task_id = request.task_id if request else None
        obs = await environment.reset(task_id=task_id)
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step_env(action: my_env.Action):
    """Process a single step decision"""
    if environment is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    try:
        obs, reward, done, info = await environment.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
async def state_env():
    """Get the current environment state tracking"""
    if environment is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    try:
        return await environment.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
