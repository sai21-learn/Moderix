import sys
import asyncio
import json
import os
import random
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

# Import graders
try:
    from graders.nsfw_grader import grade_nsfw
    from graders.reasoning_grader import grade_reasoning
    from graders.spam_grader import grade_spam
    from graders.toxicity_grader import grade_toxicity
except ImportError:
    # Fallback to dummy graders if not available yet
    def grade_toxicity(content, pred, gold):
        return 0.5

    def grade_spam(content, pred, gold):
        return 0.5

    def grade_nsfw(content, pred, gold):
        return 0.5

    def grade_reasoning(pred, gold):
        return 0.5


class Observation(BaseModel):
    """What the agent observes at each step."""

    content_id: str
    content_text: str
    source: str  # "twitter", "reddit", "discord"
    timestamp: str
    user_reputation: float = 1.0
    thread_history: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class Action(BaseModel):
    """What the agent decides to do."""

    decision: str  # "approve", "review", "reject", "escalate", "ban_user"
    violation_type: Optional[str] = (
        "none"  # "none", "toxicity", "spam", "violence", "explicit", "adult_content"
    )
    reasoning: str  # Why this decision
    confidence: float = 0.5  # 0.0-1.0


class Reward(BaseModel):
    """How the agent is scored for its action."""

    value: float = Field(ge=0.0, le=1.0)


class ContentModerationEnv:
    """OpenEnv-compliant content moderation environment."""

    def __init__(self, task_name: str = "content_moderation"):
        self.task_name = task_name
        self.current_step = 0
        self.max_steps = 8
        self.episode_rewards = []
        self.decisions_made = []
        self.gold_labels = {}
        self.current_batch = []
        self.batch_index = 0
        self.user_reputation = 1.0
        self.thread_history = []

    async def initialize(self):
        """Load gold labels and initialize."""
        print("[DEBUG] Initializing ContentModerationEnv...", file=sys.stderr)
        try:
            # Look for training_set.json in data/
            data_path = os.path.join(
                os.path.dirname(__file__), "data", "training_set.json"
            )
            print(f"[DEBUG] Attempting to load dataset from: {data_path}", file=sys.stderr)
            if not os.path.exists(data_path):
                # Try relative if above failed
                data_path = "data/training_set.json"
                print(f"[DEBUG] Path doesn't exist, falling back to: {data_path}", file=sys.stderr)

            with open(data_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
                self.gold_labels = {item["id"]: item for item in dataset}
                print(f"[INFO] Loaded {len(self.gold_labels)} gold labels", file=sys.stderr)
        except FileNotFoundError:
            print("[WARN] training_set.json not found, using empty labels", file=sys.stderr)
            self.gold_labels = {}
        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}", file=sys.stderr)
            self.gold_labels = {}
        print("[DEBUG] Initialization complete", file=sys.stderr)

    async def reset(self) -> Observation:
        """Initialize a new episode."""
        self.current_step = 0
        self.episode_rewards = []
        self.decisions_made = []
        self.user_reputation = 1.0
        self.thread_history = []

        all_posts = list(self.gold_labels.values())
        if len(all_posts) < 8:
            self.current_batch = all_posts
        else:
            # Pick 8 random posts for variety
            self.current_batch = random.sample(all_posts, 8)

        self.batch_index = 0

        if not self.current_batch:
            # Fallback: create a dummy observation
            print("[WARN] No posts in batch, returning dummy observation", file=sys.stderr)
            return Observation(
                content_id="dummy_001",
                content_text="[No content available]",
                source="test",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Return first post as observation
        post = self.current_batch[self.batch_index]
        return Observation(
            content_id=post["id"],
            content_text=post["content"],
            source=post.get("source", "twitter"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_reputation=self.user_reputation,
            thread_history=self.thread_history.copy(),
            metadata={"batch_size": len(self.current_batch)},
        )

    async def step(self, action: Action) -> tuple:
        """Process one moderation decision."""
        self.current_step += 1

        # Get gold label for this post
        current_post = (
            self.current_batch[self.batch_index]
            if self.batch_index < len(self.current_batch)
            else None
        )

        if not current_post:
            return (
                Observation(
                    content_id="end",
                    content_text="[Episode ended]",
                    source="test",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                Reward(value=0.0),
                True,
                {"error": "No more posts"},
            )

        # Grade this decision against gold labels
        reward = self._grade_decision(action, current_post)
        self.episode_rewards.append(reward)
        self.decisions_made.append(action.decision)

        # Update thread history
        self.thread_history.append(
            f"User: {current_post['content'][:50]}... -> Agent: {action.decision}"
        )

        # Move to next post
        self.batch_index += 1
        done = (self.batch_index >= len(self.current_batch)) or (
            self.current_step >= self.max_steps
        )

        # Return next observation or end signal
        if not done and self.batch_index < len(self.current_batch):
            next_post = self.current_batch[self.batch_index]
            obs = Observation(
                content_id=next_post["id"],
                content_text=next_post["content"],
                source=next_post.get("source", "twitter"),
                timestamp=datetime.now(timezone.utc).isoformat(),
                user_reputation=self.user_reputation,
                thread_history=self.thread_history[-3:],
                metadata={
                    "step": self.current_step,
                    "cumulative_reward": sum(self.episode_rewards),
                },
            )
        else:
            obs = Observation(
                content_id="done",
                content_text="[Episode complete]",
                source="test",
                timestamp=datetime.now(timezone.utc).isoformat(),
                user_reputation=self.user_reputation,
                thread_history=self.thread_history[-3:],
                metadata={"final_reward": sum(self.episode_rewards)},
            )

        info = {
            "step": self.current_step,
            "cumulative_reward": sum(self.episode_rewards),
            "decision": action.decision,
            "reasoning": action.reasoning[:50],  # truncate for logging
        }

        return obs, Reward(value=reward), done, info

    def _grade_decision(self, action: Action, post: dict) -> float:
        """Grade the agent's decision against gold labels across 4 tasks."""
        decision = action.decision.lower()
        confidence = action.confidence
        violation_type = (action.violation_type or "none").lower()

        # Check if the user is truly malicious
        is_truly_malicious = (
            post.get("toxicity", 0.0) > 0.8
            or post.get("spam", False)
            or post.get("nsfw_category", "safe")
            in ["violence", "explicit", "adult_content"]
        )

        if is_truly_malicious and decision in ["approve", "review", "escalate"]:
            self.user_reputation = max(0.0, self.user_reputation - 0.3)
        elif not is_truly_malicious and decision in ["approve"]:
            self.user_reputation = min(1.0, self.user_reputation + 0.1)

        # 1. Toxicity Grading
        # Map decision to a score (reject -> high score, approve -> low score)
        pred_tox = (
            0.9
            if decision in ["reject", "review", "ban_user"]
            and violation_type == "toxicity"
            else 0.1
        )
        if decision in ["reject", "review", "ban_user"] and violation_type == "none":
            pred_tox = 0.9  # Fallback if agent just rejects without type
        tox_reward = grade_toxicity(
            post["content"], pred_tox, post.get("toxicity", 0.0)
        )

        # 2. Spam Grading
        pred_spam = (
            True
            if decision in ["reject", "ban_user"] and violation_type == "spam"
            else False
        )
        if decision in ["reject", "ban_user"] and violation_type == "none":
            pred_spam = True  # Fallback
        spam_reward = grade_spam(post["content"], pred_spam, post.get("spam", False))

        # 3. NSFW Grading
        if decision in ["reject", "ban_user"] and violation_type in [
            "violence",
            "explicit",
            "adult_content",
        ]:
            pred_nsfw = violation_type
        elif decision in ["reject", "ban_user"] and violation_type == "nsfw":
            pred_nsfw = post.get(
                "nsfw_category", "safe"
            )  # Fallback to true category if generic nsfw
        elif decision in ["reject", "ban_user"] and violation_type == "none":
            pred_nsfw = post.get("nsfw_category", "safe")  # Fallback
        else:
            pred_nsfw = "safe"

        nsfw_reward = grade_nsfw(
            post["content"], pred_nsfw, post.get("nsfw_category", "safe")
        )

        # 4. Reasoning Quality
        reasoning_reward = grade_reasoning(
            action.reasoning, post.get("justification", "")
        )

        # Average the rewards
        base_reward = (tox_reward + spam_reward + nsfw_reward + reasoning_reward) / 4.0

        # Calibration bonus/penalty
        if base_reward > 0.7:
            final_reward = base_reward * (0.8 + 0.2 * confidence)
        else:
            final_reward = base_reward * (1.2 - 0.2 * confidence)

        # Economic / Edge-case penalties
        if decision == "review":
            final_reward -= 0.1  # Cost of human review
        elif decision == "escalate":
            final_reward -= 0.2  # Cost of legal escalation

        # Banning innocent users is catastrophic
        if decision == "ban_user" and not is_truly_malicious:
            final_reward = 0.0  # Zero out reward for false bans
            self.user_reputation = 0.0  # Break trust

        # Approving highly toxic/adversarial content is catastrophic
        if decision == "approve" and is_truly_malicious and confidence > 0.8:
            final_reward = 0.0

        # Apply trajectory penalty based on ruined reputation (min 0.2 to prevent total lock-out)
        final_reward = final_reward * max(0.2, self.user_reputation)

        return min(max(final_reward, 0.0), 1.0)

    async def state(self) -> dict:
        """Return current episode state."""
        return {
            "step": self.current_step,
            "max_steps": self.max_steps,
            "episode_rewards": self.episode_rewards,
            "decisions_made": self.decisions_made,
            "cumulative_reward": sum(self.episode_rewards),
            "batch_size": len(self.current_batch),
        }

    async def close(self):
        """Cleanup (if needed)."""
        pass

    @classmethod
    async def from_env(cls, **kwargs):
        """Initialize from environment."""
        env = cls(**kwargs)
        await env.initialize()
        return env


if __name__ == "__main__":

    async def test():
        env = await ContentModerationEnv.from_env()
        obs = await env.reset()
        print(f"Reset: {obs.content_id}, {obs.content_text[:50]}")

        action = Action(decision="approve", reasoning="Safe post", confidence=0.8)
        obs, reward, done, info = await env.step(action)
        print(f"Step 1: reward={reward}, done={done}")

        await env.close()

    asyncio.run(test())
