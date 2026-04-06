import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import asyncio
import json
import os
import re
import sys
from typing import List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Load .env file
load_dotenv()

# We import the environment locally
from my_env import Action, ContentModerationEnv

# Environment variables exactly as required by OpenEnv
API_BASE_URL = os.environ.get("API_BASE_URL", os.getenv("API_BASE_URL"))
MODEL_NAME = os.environ.get("MODEL_NAME", os.getenv("MODEL_NAME"))
HF_TOKEN = os.environ.get("HF_TOKEN", os.getenv("HF_TOKEN"))

# Fallbacks for Gemini if OpenEnv not fully set or user left placeholder token
if (not API_BASE_URL or "your_openai_or_hf_token_here" in str(HF_TOKEN)) and os.getenv(
    "GEMINI_API_KEY"
):
    API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    HF_TOKEN = os.getenv("GEMINI_API_KEY")
    if not MODEL_NAME or "gpt-4o-mini" in str(MODEL_NAME):
        MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

TASK_NAME = "content_moderation"
BENCHMARK = "openenv_moderation"
MAX_STEPS = 8

SYSTEM_PROMPT = """You are a content moderation agent. Your job is to classify and route user-generated content.

For each post, respond with EXACTLY this JSON format:
{
  "decision": "approve" | "review" | "reject" | "escalate",
  "violation_type": "none" | "toxicity" | "spam" | "nsfw" | "violence" | "explicit" | "adult_content",
  "reasoning": "Brief justification (max 100 chars)",
  "confidence": 0.5
}

Decision guide:
- "approve": Safe content, no issues (confidence > 0.8)
- "review": Moderate concern, needs human review (confidence 0.5-0.7)
- "reject": Clear policy violation, high toxicity/NSFW (confidence > 0.8)
- "escalate": Uncertain, ambiguous, needs escalation (confidence < 0.5)

Be decisive but calibrated. Follow the JSON format strictly."""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action_short: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(
        f"[END] success={str(success).lower()} steps={steps} avg_reward={avg_reward:.2f} rewards={rewards_str}",
        flush=True,
    )


def extract_json(text: str) -> str:
    """Robust JSON extraction from LLM response."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"(\{.*?\})", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=5, max=120),
    reraise=True,
)
async def _call_api(client: AsyncOpenAI, prompt: str) -> str:
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content or "{}"


async def get_model_response(client: AsyncOpenAI, content: str, step: int) -> dict:
    """Get moderation decision from OpenAI-compatible API."""
    try:
        prompt = f"Step {step}. Moderate this post:\n\n{content}"

        response_text = await _call_api(client, prompt)

        # Clean and extract JSON
        json_str = extract_json(response_text)

        try:
            parsed = json.loads(json_str)
            return {
                "decision": parsed.get("decision", "escalate"),
                "violation_type": parsed.get("violation_type", "none"),
                "reasoning": parsed.get("reasoning", "No reasoning provided"),
                "confidence": float(parsed.get("confidence", 0.5)),
            }
        except json.JSONDecodeError:
            print(
                f"[DEBUG] LLM response not JSON: {response_text[:100]}",
                flush=True,
                file=sys.stderr,
            )
            return {
                "decision": "escalate",
                "violation_type": "none",
                "reasoning": "Could not parse LLM response",
                "confidence": 0.3,
            }
    except Exception as e:
        print(f"[DEBUG] API request failed: {e}", flush=True, file=sys.stderr)
        return {
            "decision": "escalate",
            "violation_type": "none",
            "reasoning": "API error",
            "confidence": 0.2,
        }


async def main() -> None:
    if not API_BASE_URL or not MODEL_NAME or not HF_TOKEN:
        print(
            "[ERROR] API_BASE_URL, MODEL_NAME, and HF_TOKEN environment variables must be set.",
            file=sys.stderr,
        )
        return

    # Initialize OpenAI Client
    client = AsyncOpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )

    env = await ContentModerationEnv.from_env()

    rewards: List[float] = []
    step_logs: List[dict] = []
    steps_taken = 0
    success = False
    avg_reward = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = await env.reset()

        for step in range(1, MAX_STEPS + 1):
            current_content = obs.content_text

            # Get LLM decision
            response = await get_model_response(client, current_content, step)
            action = Action(
                decision=response["decision"],
                violation_type=response["violation_type"],
                reasoning=response["reasoning"][
                    :100
                ],  # truncate reasoning just in case
                confidence=response["confidence"],
            )

            # Step environment
            obs, reward_obj, done, info = await env.step(action)

            # Support both float and Reward Pydantic models
            reward = float(getattr(reward_obj, "value", reward_obj))

            rewards.append(reward)
            steps_taken = step

            step_logs.append(
                {
                    "step": step,
                    "content": current_content,
                    "action": {
                        "decision": action.decision,
                        "violation_type": action.violation_type,
                        "reasoning": action.reasoning,
                        "confidence": action.confidence,
                    },
                    "reward": reward,
                }
            )

            action_short = f"{action.decision}:{action.reasoning[:20]}"
            log_step(
                step=step,
                action_short=action_short,
                reward=reward,
                done=done,
                error=None,
            )

            if done:
                break

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        success = avg_reward >= 0.5

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True, file=sys.stderr)
        log_step(
            step=steps_taken + 1,
            action_short="error",
            reward=0.0,
            done=True,
            error=str(e),
        )
    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, rewards=rewards)

        # Save evaluation results
        results = {
            "summary": {
                "task": TASK_NAME,
                "model": MODEL_NAME,
                "success": success,
                "steps": steps_taken,
                "avg_reward": avg_reward,
            },
            "steps": step_logs,
        }
        with open("eval_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
