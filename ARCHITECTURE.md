# Architecture Overview

## Data Flow
1. **Reset**: `my_env.py` loads `data/training_set.json` and randomly samples 8 posts to form an episode.
2. **Observation**: The agent receives `Observation` with `content_text`, `source`, and `metadata`.
3. **Action**: Agent submits an `Action` containing `decision`, `reasoning`, and `confidence`.
4. **Grading**: `my_env.py` calls specific graders in `graders/` based on the ground truth of the post.
5. **Reward**: Rewards from each task are averaged and weighted by confidence to produce a scalar signal (0.0 - 1.0).
6. **Done**: Episode ends after 8 actions or if no posts remain.

## Component Breakdown
- **my_env.py**: Core environment logic, episode state management.
- **graders/**: Deterministic logic for task-specific scoring.
- **data/training_set.json**: Curated posts with multi-task labels.
- **inference.py**: Evaluation loop integrating LLM decision-making.
- **openenv.yaml**: OpenEnv compliance specification.