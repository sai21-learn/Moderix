import sys
import yaml
from fastapi import FastAPI, HTTPException
import my_env
from pydantic import BaseModel

app = FastAPI(title="Content Moderation OpenEnv API")

# Global environment instance
environment: my_env.ContentModerationEnv = None
_init_error: str = None

# Load tasks from openenv.yaml at startup
_TASKS = []
try:
    with open("openenv.yaml") as f:
        _cfg = yaml.safe_load(f)
        _TASKS = _cfg.get("tasks", [])
except Exception as e:
    print(f"[WARN] Could not load openenv.yaml: {e}", file=sys.stderr, flush=True)

@app.on_event("startup")
async def startup_event():
    global environment, _init_error
    try:
        environment = await my_env.ContentModerationEnv.from_env()
        print("[INFO] Environment initialized successfully", file=sys.stderr, flush=True)
    except Exception as e:
        _init_error = str(e)
        print(f"[ERROR] Environment initialization failed: {e}", file=sys.stderr, flush=True)
        try:
            environment = my_env.ContentModerationEnv()
        except Exception:
            pass

@app.get("/")
async def health_check():
    """Hugging Face Spaces automated ping / healthcheck endpoint - always returns 200."""
    status = "ready" if environment is not None else "initializing"
    return {"status": status, "error": _init_error}

@app.get("/health")
async def health_detailed():
    """Detailed health endpoint."""
    return {"status": "healthy", "env_ready": environment is not None}

# ── Task enumeration endpoint ────────────────────────────────────────────────
@app.get("/tasks")
async def list_tasks():
    """
    Enumerate all available tasks with their graders.
    Required by the OpenEnv evaluator to detect distinct tasks.
    """
    return {
        "tasks": [
            {
                "id":          t.get("id"),
                "name":        t.get("name"),
                "description": t.get("description"),
                "difficulty":  t.get("difficulty"),
                "grader":      t.get("grader"),
            }
            for t in _TASKS
        ],
        "count": len(_TASKS),
    }

# ── Per-task grading endpoint ─────────────────────────────────────────────────
@app.post("/grade/{task_id}")
async def grade_task(task_id: str, observation: dict, action: dict):
    """
    Run the grader for a specific task and return the reward.
    Allows the evaluator to verify grading logic per task.
    """
    import importlib.util, os

    # Find matching task
    task = next((t for t in _TASKS if t["id"] == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    grader_path = task.get("grader")
    if not grader_path or not os.path.exists(grader_path):
        raise HTTPException(status_code=404, detail=f"Grader '{grader_path}' not found")

    try:
        spec = importlib.util.spec_from_file_location("grader_module", grader_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        reward = mod.grade(observation, action)
        return {"task_id": task_id, "reward": float(reward)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Episode endpoints ─────────────────────────────────────────────────────────
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
            "reward":      reward.model_dump(),
            "done":        done,
            "info":        info,
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
