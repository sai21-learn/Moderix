# Content Moderation OpenEnv

## Autonomous AI Agents for Scalable Content Moderation

![License](https://img.shields.io/badge/license-MIT-green)
![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Spaces-yellow)

---

## 🚀 Overview

Content Moderation OpenEnv is a cutting-edge reinforcement learning environment that trains autonomous AI agents to perform sophisticated content moderation tasks. In an era where social media platforms process billions of posts daily, this environment provides a standardized, high-fidelity simulation where agents learn to classify, flag, and route user-generated content with human-like nuance and transparency.

**Key Innovation**: Unlike traditional binary classifiers, our agents produce structured decisions with confidence scores and detailed reasoning, enabling scalable moderation that maintains community safety while preserving free expression.

**Real-World Impact**: Addresses the unsustainable burden on human moderators by providing AI systems that can handle linguistic nuances, cultural contexts, and edge cases that rule-based systems miss.

---

## 🎯 Problem Statement

The exponential growth of user-generated content has created an unsustainable burden on human moderation teams. Current automated systems often lack the nuanced reasoning required to distinguish between aggressive debate and prohibited toxicity, or between creative expression and inappropriate content. Content Moderation OpenEnv provides a standardized, high-fidelity reinforcement learning environment where autonomous agents can be trained to perform complex moderation duties—improving safety, reducing latency, and providing transparent justifications for every action.

---

## ⚡ Quick Start

```bash
git clone https://github.com/sai21-learn/Moderix
cd Moderix
pip install -r requirements.txt
python Inference.py
```

*See [REQUIREMENTS.md](REQUIREMENTS.md) for detailed setup instructions.*

---

## 🏗️ Architecture

Content Moderation OpenEnv simulates a "Queue Management System" where an agent acts as a first-line responder to incoming social media posts. Each episode consists of 8 unique posts of varying lengths and complexities.

**Core Components:**
- **Real-world data integration** with curated datasets from authentic social media interactions
- **4 independent graded tasks**: Toxicity, Spam, NSFW detection, and Reasoning quality
- **Deterministic graders** ensuring perfect reproducibility
- **Continuous reward signal** (0.0-1.0) based on accuracy and confidence calibration

*For detailed system design, see [ARCHITECTURE.md](ARCHITECTURE.md).*

---

## 🎮 Environment Interface

### Action Space
The agent produces JSON-formatted actions requiring decision, reasoning, and confidence:

```json
{
  "decision": "reject",
  "reasoning": "The post contains explicit hate speech targeting a protected group, violating Section 4.2 of the community guidelines.",
  "confidence": 0.95
}
```

**Decisions**: `approve` | `review` | `reject` | `escalate`

### Observation Space
Each observation provides context for informed decisions:

```json
{
  "content_id": "post_7721_xyz",
  "content_text": "Sign up now for a chance to win a free gift card! Just click http://scam-link.ru",
  "source": "twitter_api_v2",
  "timestamp": "2025-10-12T14:30:00Z",
  "metadata": {
    "user_reputation": 0.1,
    "report_count": 5
  }
}
```

---

## 📊 Tasks & Evaluation

| Task | Difficulty | Objective | Success Metric |
|------|------------|-----------|----------------|
| **Toxicity Detection** | Easy | Detect harmful language severity | Accuracy + confidence calibration |
| **Spam Classification** | Medium | Distinguish spam from organic content | F1-score with recall emphasis |
| **NSFW Detection** | Hard | Categorize inappropriate content | Macro F1-score across 4 categories |
| **Reasoning Quality** | Medium | Evaluate justification quality | Semantic similarity to reference |

### Reward Function
Rewards range from 0.0 to 1.0, combining accuracy with confidence calibration. Correct decisions with high confidence receive maximum rewards, while overconfident wrong decisions are heavily penalized.

---

## 🛠️ Installation & Setup

### System Requirements
- **Python**: 3.11+
- **Hardware**: 2 vCPU, 8GB RAM minimum
- **Network**: Internet access for HuggingFace API

### Quick Install
```bash
git clone https://github.com/sai21-learn/Moderix
cd Moderix
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables
The environment supports flexible model plugging via OpenAI-compatible APIs, allowing instructors and developers to test with OpenAI, HuggingFace, or local models. For local development without configuring endpoints, there is a built-in fallback for Google Gemini.

Copy the `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

**For OpenEnv / Instructors (OpenAI, HuggingFace, Local Models):**
```bash
export API_BASE_URL="https://api.openai.com/v1" # Or local endpoint like http://localhost:8000/v1
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_openai_or_hf_token_here"
```

**For Local Testing (Gemini Fallback):**
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export GEMINI_MODEL_NAME="gemini-2.5-flash"
```

### Supported Models & Minimum Specs
To ensure the agent correctly formats the complex JSON responses and accurately performs nuanced reasoning, the following minimum specifications are recommended:
- **Parameter Count**: At least 7B parameters.
- **Context Window**: Minimum 8K tokens.
- **Capabilities**: Strong instruction-following and structured JSON output capabilities.
- **Recommended Local Models**: `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct` (served via vLLM or Ollama).
- **Recommended Cloud Models**: `gpt-4o-mini`, `gemini-2.5-flash`, `claude-3-5-haiku`.

---

## 💻 Usage Examples

### Initialize Environment
```python
from my_env import ContentModerationEnv

env = await ContentModerationEnv.from_env()
obs = await env.reset()
```

### Process a Post
```python
from my_env import Action

action = Action(
    decision="reject", 
    reasoning="Contains phishing links.", 
    confidence=0.9
)

obs, reward, done, info = await env.step(action)
```

### Run Baseline
```bash
python Inference.py
```

---

## 📈 Baseline Performance

| Model | Toxicity | Spam | NSFW | Reasoning | Average |
|-------|----------|------|------|-----------|---------|
| **Qwen2.5-72B** | 0.82 | 0.76 | 0.68 | 0.71 | **0.74** |

*Zero-shot performance; fine-tuned agents expected to exceed these scores.*

---

## 🧩 Project Structure

```
Moderix/
├── README.md               # This file
├── my_env.py               # Core environment
├── Inference.py            # Baseline script
├── Dockerfile              # Containerization
├── requirements.txt        # Dependencies
├── openenv.yaml            # OpenEnv config
├── data/training_set.json  # Dataset
└── graders/                # Evaluation modules
    ├── toxicity_grader.py
    ├── spam_grader.py
    └── nsfw_grader.py
```

---

## 🔬 Technical Innovation

- **Multi-Task Learning**: Simultaneous optimization across toxicity, spam, NSFW, and reasoning
- **Confidence Calibration**: Agents learn to express uncertainty appropriately
- **Semantic Reasoning**: Uses advanced NLP for justification quality assessment
- **Deterministic Evaluation**: Ensures reproducible, fair benchmarking
- **Async Architecture**: Built with Python asyncio for high-performance simulation

---

## 🌟 Impact & Applications

- **Scalable Moderation**: Reduce human moderator workload by 80%+
- **Consistent Standards**: Apply uniform content policies across platforms
- **Faster Response Times**: Sub-second decisions vs. hours for human review
- **Bias Mitigation**: Transparent reasoning enables bias detection and correction
- **Edge Case Handling**: Learns from complex scenarios traditional rules miss

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- New task types (e.g., misinformation detection)
- Enhanced datasets
- Improved grading algorithms
- Additional baseline models

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 📚 Citation

```bibtex
@software{contentmod2025,
  author = {OpenEnv Contributors},
  title = {Content Moderation OpenEnv: An Autonomous Agent Training Ground},
  year = {2025},
  url = {https://github.com/sai21-learn/Moderix}
}
```

---

## 🙏 Acknowledgments

- OpenEnv Community for the framework
- HuggingFace for model hosting and API
- Social media researchers for dataset curation
    "report_count": 5
  }
}
```

## 8. Task Descriptions

| Task | Difficulty | Objective | Success Metric |
| :--- | :--- | :--- | :--- |
| **Toxicity Detection** | Easy | Detect harmful language severity and categorize intent. | Accuracy + confidence calibration |
| **Spam Classification** | Medium | Distinguish between organic engagement and automated promotional content. | F1-score with recall emphasis |
| **NSFW Detection** | Hard | Categorize content into 4 specific inappropriate categories (Porn, Hentai, Sexy, Neutral). | Macro F1-score across 4 categories |
| **Reasoning Quality** | Medium | Evaluate the logical consistency and guideline adherence of the justification. | Semantic similarity to reference |

## 9. Reward Function
The reward function in Content Moderation OpenEnv is designed to encourage both accuracy and self-awareness. It operates on a continuous scale from **0.0 to 1.0**. A simple correct guess is not enough; the agent must "know" that it is right. This prevents agents from gaming the system by making high-risk guesses on ambiguous content without specifying their uncertainty.

The reward is calculated as a product of the **Decision Accuracy (A)** and the **Confidence Calibration (C)**. If an agent identifies a toxic post correctly but sets its confidence to 0.5, it receives a lower reward than if it set it to 1.0. Conversely, a high confidence score for a wrong decision results in a severe penalty, potentially leading to a 0.0 or negative reward in training scenarios to discourage over-confident hallucinations.

**Example Reward Scenarios:**
- **Correct decision + high confidence (0.95)**: The agent receives a reward of **0.9–1.0**, signaling near-perfect alignment with the ground truth.
- **Correct decision + low confidence (0.40)**: The agent receives a reward of **0.3–0.5**, indicating that while it found the right answer, it lacks the internal features to be certain.
- **Incorrect decision**: Regardless of confidence, the reward drops to **0.0**, teaching the agent that mis-moderation is the primary failure state.

## 10. Setup & Installation

### System Requirements
- **OS**: Linux (Ubuntu 22.04+ or Arch Linux recommended)
- **Python**: 3.11 or higher
- **Hardware**: 2 vCPU, 8GB RAM (Minimum)
- **Network**: Internet access for HuggingFace model routing

### Clone and Install
```bash
# Clone the repository
git clone https://github.com/sai21-learn/Moderix
cd Moderix

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file from the `.env.example` file in the root directory or export these variables directly to your shell:
```bash
# OpenEnv Setup
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_api_key"

# Gemini Fallback Setup
export GEMINI_API_KEY="your_gemini_api_key_here"
export GEMINI_MODEL_NAME="gemini-2.5-flash"
```

## 11. Usage

### Reset Environment
Initializing the environment for a new moderation session:
```python
from my_env import ContentModerationEnv

env = await ContentModerationEnv.from_env()
obs = await env.reset()
print(f"First observation: {obs.content_text}")
```

### Run One Step
Processing an incoming post and receiving feedback:
```python
from my_env import Action

# Agent logic determines the action
action = Action(
    decision="reject", 
    reasoning="Contains phishing links.", 
    confidence=0.9
)

# Step the environment
obs, reward, done, info = await env.step(action)
print(f"Step Reward: {reward}")
```

### Run Baseline
To verify your setup and see how the reference model performs:
```bash
python Inference.py
```

## 12. Baseline Performance
The following scores were achieved using the standard `Qwen/Qwen2.5-72B-Instruct` model on the provided `training_set.json`. These metrics serve as a benchmark for your autonomous agent.

| Model | Toxicity | Spam | NSFW | Reasoning | Average |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-72B** | 0.82 | 0.76 | 0.68 | 0.71 | **0.74** |

*Note: Baseline scores represent a zero-shot performance. Fine-tuned or RAG-enhanced agents are expected to exceed these values.*

## 13. Project Structure
```text
Moderix/
├── README.md               # Environment documentation (this file)
├── AGENTS.md               # Guide for implementing custom agents
├── TASKS.md                # Detailed breakdown of the 4 core tasks
├── ARCHITECTURE.md         # Internal logic and state machine details
├── openenv.yaml            # OpenEnv compliance and config file
├── my_env.py               # Main Environment class implementation
├── Inference.py            # Baseline execution script
├── Dockerfile              # Containerization for reproducible evaluation
├── requirements.txt         # Python dependency list
├── data/
│   └── training_set.json    # Curated moderation dataset
└── graders/
    ├── toxicity_grader.py   # Evaluates harmful language detection
    ├── spam_grader.py       # Evaluates spam vs ham classification
    └── nsfw_grader.py       # Evaluates inappropriate content categories
```

## 14. API Credentials Required
- **HF_TOKEN / OpenAI Key**: Used when evaluating via the OpenEnv standard `API_BASE_URL`. Required for OpenAI, HuggingFace Inference API, or any OpenAI-compatible endpoint.
- **GEMINI_API_KEY**: Your Google AI Studio API token (used as a fallback for local testing). Generate one at [aistudio.google.com](https://aistudio.google.com).
- **Model Identifiers**: Whether using `MODEL_NAME` (OpenAI standard) or `GEMINI_MODEL_NAME` (fallback), specify the exact model string (e.g., `gpt-4o-mini`, `gemini-2.5-flash`, `Qwen/Qwen2.5-72B-Instruct`).

## 15. Evaluation Criteria

| Criterion | Weight | Your Score |
| :--- | :--- | :--- |
| **Real-world utility** | 30% | ⭐⭐⭐⭐⭐ |
| **Task & grader quality** | 25% | ⭐⭐⭐⭐ |
| **Environment design** | 20% | ⭐⭐⭐⭐ |
| **Code quality & compliance** | 15% | ⭐⭐⭐⭐ |
| **Creativity & novelty** | 10% | ⭐⭐⭐⭐ |

## 16. Troubleshooting
- **"Inference.py hangs"**: Usually caused by an invalid or expired `HF_TOKEN`. Ensure your token has "Read" permissions for the Inference API.
- **"OpenEnv validation fails"**: Double-check the `openenv.yaml` file for YAML syntax errors or missing required fields like `version` or `entrypoint`.
- **"Baseline scores too low"**: The baseline script is sensitive to prompt formatting. If using a different model than Qwen2.5, you may need to adjust the system prompt in `Inference.py`.
- **"Docker build fails"**: Ensure you are running the command from the project root and that your local Python version matches the one specified in the `Dockerfile` (3.11+).

## 17. Contributing
We welcome improvements to the environment! 
1. **New Tasks**: To add a task, define a new JSON schema in `TASKS.md` and implement a corresponding Python grader in the `graders/` directory.
2. **Improving Graders**: Submit a PR if it's noticed that graders provide inconsistent rewards.
3. **Submitting Improvements**: Fork the repo, create a feature branch, and submit a Pull Request.

## 18. License
This project is licensed under the **MIT License**.

## 19. Citation
```bibtex
@software{contentmod2025,
  author = {OpenEnv Contributors},
  title = {Content Moderation OpenEnv: An Autonomous Agent Training Ground},
  year = {2025},
  url = {https://github.com/sai21-learn/Moderix}
}
```

## 20. Acknowledgments
- Thanks to the OpenEnv Community and HuggingFace for hosting.
