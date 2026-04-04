# inference.py
import asyncio
import os
import json
import re
from typing import List, Optional
from openai import OpenAI
from my_env import ContentModerationEnv, Action
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

TASK_NAME = "content_moderation"
BENCHMARK = "openenv_moderation"
MAX_STEPS = 8

SYSTEM_PROMPT = """You are a content moderation agent. Your job is to classify and route user-generated content.

For each post, respond with EXACTLY this JSON format:
{
  "decision": "approve" | "review" | "reject" | "escalate",
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

def log_step(step: int, action_short: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"[END] success={str(success).lower()} steps={steps} avg_reward={avg_reward:.2f} rewards={rewards_str}", flush=True)

def extract_json(text: str) -> str:
    """Robust JSON extraction from LLM response."""
    # Try looking for JSON code blocks
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Fallback: find anything that looks like a JSON object
    match = re.search(r'(\{.*?\})', text, re.DOTALL)
    if match:
        return match.group(1)
    
    return text

def get_model_response(client: OpenAI, model_name: str, content: str, step: int) -> dict:
    """Get moderation decision from Gemini via OpenAI SDK."""
    try:
        prompt = f"Step {step}. Moderate this post:\n\n{content}"
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        response_text = response.choices[0].message.content or "{}"
        
        # Clean and extract JSON
        json_str = extract_json(response_text)
            
        # Try to parse JSON
        try:
            parsed = json.loads(json_str)
            return {
                "decision": parsed.get("decision", "escalate"),
                "reasoning": parsed.get("reasoning", "No reasoning provided"),
                "confidence": float(parsed.get("confidence", 0.5))
            }
        except json.JSONDecodeError:
            print(f"[DEBUG] LLM response not JSON: {response_text[:100]}", flush=True)
            return {
                "decision": "escalate",
                "reasoning": "Could not parse LLM response",
                "confidence": 0.3
            }
    except Exception as e:
        print(f"[DEBUG] Gemini request failed: {e}", flush=True)
        return {
            "decision": "escalate",
            "reasoning": "API error",
            "confidence": 0.2
        }

async def main() -> None:
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY environment variable not set.")
        return

    # Initialize OpenAI Client for Gemini
    client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    env = await ContentModerationEnv.from_env()
    
    rewards: List[float] = []
    steps_taken = 0
    success = False
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # Reset and get first observation
        obs = await env.reset()
        
        for step in range(1, MAX_STEPS + 1):
            # Get LLM decision
            response = get_model_response(client, MODEL_NAME, obs.content_text, step)
            action = Action(
                decision=response["decision"],
                reasoning=response["reasoning"],
                confidence=response["confidence"]
            )
            
            # Step environment
            obs, reward, done, info = await env.step(action)
            
            rewards.append(reward)
            steps_taken = step
            
            action_short = f"{action.decision}:{action.reasoning[:20]}"
            log_step(step=step, action_short=action_short, reward=reward, done=done, error=None)
            
            if done:
                break
            
            # Avoid Gemini free tier rate limit: 5 requests per minute
            await asyncio.sleep(15)
        
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        success = avg_reward >= 0.5  # Threshold for success
        
    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
