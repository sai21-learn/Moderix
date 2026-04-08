"""
Moderix – OpenEnv inference script
===================================
Implementation following the strict structured logging spec.
"""

import os
import json
import time
import textwrap
import sys
from typing import List, Optional

from openai import OpenAI
import httpx

# ── Configuration ───────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

API_KEY = OPENAI_API_KEY or HF_TOKEN

SERVER_URL = os.getenv("OPENENV_SERVER_URL", "http://localhost:7860")
# Read TASK_ID from environment (standard for OpenEnv validators)
TASK_NAME  = os.getenv("TASK_ID", os.getenv("TASK", os.getenv("TASK_NAME", "toxicity_detection")))
BENCHMARK = "moderix"
MAX_STEPS = 8
# Theoretical max reward per step is 1.0 (perfect prediction across all categories)
MAX_TOTAL_REWARD = float(MAX_STEPS)

SYSTEM_PROMPT = textwrap.dedent("""
    You are a content moderation agent. Respond with strict JSON:
    {
      "decision": "approve" | "review" | "reject" | "escalate" | "ban_user",
      "violation_type": "none" | "toxicity" | "spam" | "nsfw" | "violence",
      "reasoning": "string",
      "confidence": 0.0-1.0
    }
""").strip()

# ── Structured Logging ──────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── Helpers ─────────────────────────────────────────────────────────────────
def get_model_decision(client: OpenAI, content: str) -> dict:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Moderate this: {content}"},
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {
            "decision": "escalate",
            "violation_type": "none",
            "reasoning": f"Error: {str(e)}",
            "confidence": 0.0
        }

async def main() -> None:
    # Build Client
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    async with httpx.AsyncClient(timeout=30.0, base_url=SERVER_URL) as http:
        # Wait for server
        for _ in range(30):
            try:
                r = await http.get("/health")
                if r.status_code == 200:
                    break
            except:
                pass
            await time.sleep(2)

        try:
            # Episode Loop
            # Pass task_id so environment can focus rewards correctly
            r = await http.post("/reset", json={"task_id": TASK_NAME})
            obs = r.json()
            
            for step in range(1, MAX_STEPS + 1):
                content = obs.get("content_text", "")
                decision = get_model_decision(client, content)
                
                # Step
                r = await http.post("/step", json=decision)
                result = r.json()
                
                obs = result.get("observation", {})
                reward = float(result.get("reward", {}).get("value", 0.0))
                done = bool(result.get("done", False))
                
                rewards.append(reward)
                steps_taken = step
                
                action_str = f"{decision.get('decision')}:{decision.get('violation_type')}"
                log_step(step=step, action=action_str, reward=reward, done=done, error=None)

                if done:
                    break
            
            score = sum(rewards) / MAX_TOTAL_REWARD
            success = score >= 0.5

        except Exception as e:
            log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(e))
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
