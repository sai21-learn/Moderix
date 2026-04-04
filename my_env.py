import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field
import json

class Observation(BaseModel):
    """What the agent observes at each step."""
    content_id: str
    content_text: str
    source: str  # "twitter", "reddit", "discord"
    timestamp: str
    metadata: dict = Field(default_factory=dict)

class Action(BaseModel):
    """What the agent decides to do."""
    decision: str  # "approve", "review", "reject", "escalate"
    reasoning: str  # Why this decision
    confidence: float = 0.5  # 0.0-1.0

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
        
    async def initialize(self):
        """Load gold labels and initialize."""
        ### STEP 1: Load training_set.json from data/
        try:
            with open("data/training_set.json", "r") as f:
                dataset = json.load(f)
                self.gold_labels = {item["id"]: item for item in dataset}
                print(f"[INFO] Loaded {len(self.gold_labels)} gold labels")
        except FileNotFoundError:
            print("[WARN] training_set.json not found, using empty labels")
            self.gold_labels = {}
    
    async def reset(self) -> Observation:
        """Initialize a new episode."""
        self.current_step = 0
        self.episode_rewards = []
        self.decisions_made = []
        
        ### STEP 2: Load a batch of posts (8 posts from gold_labels)
        posts = list(self.gold_labels.values())[:8]
        self.current_batch = posts
        self.batch_index = 0
        
        if not self.current_batch:
            # Fallback: create a dummy observation
            print("[WARN] No posts in batch, returning dummy observation")
            return Observation(
                content_id="dummy_001",
                content_text="[No content available]",
                source="test",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        # Return first post as observation
        post = self.current_batch[self.batch_index]
        return Observation(
            content_id=post["id"],
            content_text=post["content"],
            source=post.get("source", "twitter"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={"batch_size": len(self.current_batch)}
        )
    
    async def step(self, action: Action) -> tuple:
        """Process one moderation decision."""
        self.current_step += 1
        
        # Get gold label for this post
        current_post = self.current_batch[self.batch_index] if self.batch_index < len(self.current_batch) else None
        
        if not current_post:
            return Observation(
                content_id="end",
                content_text="[Episode ended]",
                source="test",
                timestamp=datetime.now(timezone.utc).isoformat()
            ), 0.0, True, {"error": "No more posts"}
        
        ### STEP 3: Grade this decision against gold labels
        reward = self._grade_decision(action, current_post)
        self.episode_rewards.append(reward)
        self.decisions_made.append(action.decision)
        
        # Move to next post
        self.batch_index += 1
        done = (self.batch_index >= len(self.current_batch)) or (self.current_step >= self.max_steps)
        
        # Return next observation or end signal
        if not done and self.batch_index < len(self.current_batch):
            next_post = self.current_batch[self.batch_index]
            obs = Observation(
                content_id=next_post["id"],
                content_text=next_post["content"],
                source=next_post.get("source", "twitter"),
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"step": self.current_step, "cumulative_reward": sum(self.episode_rewards)}
            )
        else:
            obs = Observation(
                content_id="done",
                content_text="[Episode complete]",
                source="test",
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"final_reward": sum(self.episode_rewards)}
            )
        
        info = {
            "step": self.current_step,
            "cumulative_reward": sum(self.episode_rewards),
            "decision": action.decision,
            "reasoning": action.reasoning[:50]  # truncate for logging
        }
        
        return obs, reward, done, info
    
    def _grade_decision(self, action: Action, post: dict) -> float:
        """Grade the agent's decision against gold labels."""
        ### STEP 4: Implement grading logic
        
        decision = action.decision.lower()
        confidence = action.confidence
        
        # Simple accuracy-based grading
        # Toxicity task
        if "toxicity" in post:
            gold_tox = post["toxicity"]
            if gold_tox > 0.5:  # Toxic post
                if decision in ["review", "reject"]:
                    reward = 0.8 + 0.2 * min(confidence, 0.5)
                else:
                    reward = 0.1
            else:  # Safe post
                if decision == "approve":
                    reward = 0.9 + 0.1 * confidence
                else:
                    reward = 0.3
        else:
            reward = 0.5  # Neutral if no gold label
        
        return min(max(reward, 0.0), 1.0)  # Clamp to [0, 1]
    
    async def state(self) -> dict:
        """Return current episode state."""
        return {
            "step": self.current_step,
            "max_steps": self.max_steps,
            "episode_rewards": self.episode_rewards,
            "decisions_made": self.decisions_made,
            "cumulative_reward": sum(self.episode_rewards),
            "batch_size": len(self.current_batch)
        }
    
    async def close(self):
        """Cleanup (if needed)."""
        pass
    
    @classmethod
    async def from_docker_image(cls, image_name: str):
        """For HF Space compatibility."""
        env = cls()
        await env.initialize()
        return env
    
    @classmethod
    async def from_env(cls, **kwargs):
        """Initialize from environment variables."""
        env = cls(**kwargs)
        await env.initialize()
        return env