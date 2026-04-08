import sys
import os
import logging
# Absolute suppression of all noisy AI logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

# Suppress HuggingFace warnings and force offline if model is pre-loaded
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Model is lazy-loaded to avoid blocking startup
model = None
HAS_MODEL = True

def get_model():
    global model, HAS_MODEL
    if model is None and HAS_MODEL:
        try:
            # Suppress transformers/hf hub logs during load
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            
            from sentence_transformers import SentenceTransformer
            
            # Use local path if it exists (for Docker pre-loaded models)
            local_path = os.path.join(os.environ.get("HOME", "/home/user"), ".cache/torch/sentence_transformers/all-MiniLM-L6-v2")
            model_name = local_path if os.path.exists(local_path) else "all-MiniLM-L6-v2"
            
            model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"[WARN] Failed to load SentenceTransformer: {e}", file=sys.stderr)
            HAS_MODEL = False
    return model


def grade_reasoning(predicted_reasoning: str, gold_justification: str) -> float:
    """
    Grade reasoning quality using semantic similarity (sentence-transformers).
    Falls back to difflib if sentence-transformers is not available.

    Args:
        predicted_reasoning: Agent's reasoning string
        gold_justification: True justification

    Returns:
        Reward (0.0-1.0)
    """
    pred = predicted_reasoning.strip()
    gold = gold_justification.strip()

    if not pred or not gold:
        return 0.1

    current_model = get_model()
    if HAS_MODEL and current_model is not None:
        try:
            import numpy as np
            # Compute embeddings
            embeddings = current_model.encode([pred, gold])

            # Compute cosine similarity
            norm_0 = np.linalg.norm(embeddings[0])
            norm_1 = np.linalg.norm(embeddings[1])

            if norm_0 == 0 or norm_1 == 0:
                return 0.1

            sim = np.dot(embeddings[0], embeddings[1]) / (norm_0 * norm_1)
            ratio = float(max(0.0, sim))
        except Exception as e:
            print(f"[WARN] Error in reasoning grader: {e}", file=sys.stderr)
            ratio = 0.0
    else:
        # Fallback to simple difflib
        import difflib
        pred_lower = pred.lower()
        gold_lower = gold.lower()
        seq = difflib.SequenceMatcher(None, pred_lower, gold_lower)
        ratio = seq.ratio()

    # Smooth the curve
    if ratio > 0.7:
        return 1.0
    elif ratio > 0.5:
        return 0.8
    elif ratio > 0.3:
        return 0.5
    elif ratio > 0.1:
        return 0.2
    else:
        return 0.0


if __name__ == "__main__":
    r1 = grade_reasoning("Contains toxic insults targeting a group", "Toxic insults")
    print(f"Good match: {r1}")
    r2 = grade_reasoning("Safe and wholesome content", "Toxic insults")
    print(f"Bad match: {r2}")
