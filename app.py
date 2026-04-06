import asyncio
from fastapi import FastAPI, HTTPException
import my_env
from pydantic import BaseModel

app = FastAPI(title="Content Moderation OpenEnv API")

# Global environment instance
environment: my_env.ContentModerationEnv = None

@app.on_event("startup")
async def startup_event():
    global environment
    # Initialize the Content Moderation RL environment exactly as Inference.py does
    environment = await my_env.ContentModerationEnv.from_env()

@app.get("/")
async def health_check():
    """Hugging Face Spaces automated ping endpoint - must return 200"""
    if environment is not None:
        return {"status": "healthy", "message": "Environment is ready"}
    raise HTTPException(status_code=503, detail="Environment initializing...")

@app.post("/reset")
async def reset_env():
    """Reset the environment state for a new episode"""
    try:
        obs = await environment.reset()
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step_env(action: my_env.Action):
    """Process a single step decision"""
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
    try:
        return await environment.state()
    except getattr(Exception, "dummy", Exception) as e:
        raise HTTPException(status_code=500, detail=str(e))
