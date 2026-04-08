"""
Moderix – OpenEnv inference script
===================================
MANDATORY: Exactly follows the [START]/[STEP]/[END] log spec.

Env vars required:
  API_BASE_URL  - LLM API endpoint
  MODEL_NAME    - model identifier
  HF_TOKEN      - Hugging Face / API key
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

SERVER_URL = os.getenv("OPENENV_SERVER_URL", "http://localhost:7860")

# Task identifier – the validator will inject this via an env var
TASK_NAME  = os.getenv("TASK_ID") or os.getenv("TASK") or os.getenv("TASK_NAME") or "toxicity_detection"
BENCHMARK  = "moderix"

MAX_STEPS             = 8
SUCCESS_SCORE_THRESHOLD = 0.3   # ≥30% is a pass
# Each step scores 0-1, so max total = MAX_STEPS * 1.0
MAX_TOTAL_REWARD = float(MAX_STEPS)

SYSTEM_PROMPT = textwrap.dedent("""
    You are a content moderation agent. Given a post or comment, respond
    in strict JSON with exactly these keys:
    {
      "decision":       "approve" | "review" | "reject" | "escalate" | "ban_user",
      "violation_type": "none" | "toxicity" | "spam" | "nsfw" | "violence",
      "reasoning":      "<one sentence explaining why>",
      "confidence":     <float 0.0-1.0>
    }
    Respond with ONLY the JSON object and nothing else.
""").strip()

# ── Structured Logging ───────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM Helper ───────────────────────────────────────────────────────────────
def get_model_decision(client: OpenAI, content: str) -> dict:
    """Call the LLM and parse its JSON response. Always returns a valid dict."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Moderate this content:\n\n{content}"},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return {
            "decision":       "escalate",
            "violation_type": "none",
            "reasoning":      f"LLM error: {str(exc)[:100]}",
            "confidence":     0.0,
        }


# ── Server Readiness ─────────────────────────────────────────────────────────
async def wait_for_server(http: httpx.AsyncClient, retries: int = 30, delay: float = 2.0) -> bool:
    """Poll /health until the environment server responds."""
    for attempt in range(retries):
        try:
            r = await http.get("/health", timeout=5.0)
            if r.status_code == 200:
                print(f"[DEBUG] Server ready after {attempt * delay:.0f}s", flush=True)
                return True
        except Exception:
            pass
        print(f"[DEBUG] Waiting for server... ({attempt + 1}/{retries})", flush=True)
        await asyncio.sleep(delay)
    return False


# ── Main Loop ────────────────────────────────────────────────────────────────
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Determine tasks to run: if a specific task is passed, run only that.
    # Otherwise, run all 3 benchmark tasks so the evaluator sees them all.
    run_specific = os.getenv("TASK_ID") or os.getenv("TASK") or os.getenv("TASK_NAME")
    tasks_to_run = [run_specific] if run_specific else ["toxicity_detection", "spam_classification", "nsfw_detection"]

    async with httpx.AsyncClient(base_url=SERVER_URL, timeout=45.0) as http:
        # Wait for env server ONCE before looping
        server_up = await wait_for_server(http, retries=40, delay=2.0)
        if not server_up:
            print("[DEBUG] Server did not become ready in time", flush=True)
            for t in tasks_to_run:
                log_start(task=t, env=BENCHMARK, model=MODEL_NAME)
                log_end(success=False, steps=0, score=0.0, rewards=[])
            return

        for current_task in tasks_to_run:
            rewards:     List[float] = []
            steps_taken: int         = 0
            score:       float       = 0.0
            success:     bool        = False

            log_start(task=current_task, env=BENCHMARK, model=MODEL_NAME)

            try:
                # ── Reset episode for specific task ─────────
                r = await http.post("/reset", json={"task_id": current_task})
                r.raise_for_status()
                obs = r.json()

                # ── Step loop ───────────────────────────────
                for step in range(1, MAX_STEPS + 1):
                    content = obs.get("content_text", "[no content]")

                    # Ask the LLM
                    decision = get_model_decision(client, content)

                    # Send action to environment
                    r = await http.post("/step", json=decision)
                    r.raise_for_status()
                    result = r.json()

                    obs     = result.get("observation", {})
                    
                    # Sometimes reward is an object {"value": 0.5}, sometimes just a float
                    raw_reward = result.get("reward", 0.0)
                    if isinstance(raw_reward, dict):
                        reward = float(raw_reward.get("value", 0.0))
                    else:
                        reward = float(raw_reward)
                        
                    done    = bool(result.get("done", False))

                    rewards.append(reward)
                    steps_taken = step

                    action_str = f"{decision.get('decision', 'unknown')}:{decision.get('violation_type', 'none')}"
                    log_step(step=step, action=action_str, reward=reward, done=done, error=None)

                    if done:
                        break

                score   = min(max(sum(rewards) / MAX_TOTAL_REWARD, 0.0), 1.0)
                success = score >= SUCCESS_SCORE_THRESHOLD

            except Exception as exc:
                print(f"[DEBUG] Episode error on task {current_task}: {exc}", flush=True)
                log_step(
                    step=steps_taken + 1,
                    action="error",
                    reward=0.0,
                    done=True,
                    error=str(exc)[:200],
                )
            finally:
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
