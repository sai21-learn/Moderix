import difflib


def grade_reasoning(predicted_reasoning: str, gold_justification: str) -> float:
    """
    Grade reasoning quality using semantic similarity (SequenceMatcher as proxy).

    Args:
        predicted_reasoning: Agent's reasoning string
        gold_justification: True justification

    Returns:
        Reward (0.0-1.0)
    """
    pred = predicted_reasoning.lower().strip()
    gold = gold_justification.lower().strip()

    seq = difflib.SequenceMatcher(None, pred, gold)
    ratio = seq.ratio()

    # Smooth the curve
    if ratio > 0.5:
        return 1.0
    elif ratio > 0.3:
        return 0.7
    elif ratio > 0.1:
        return 0.3
    else:
        return 0.1


if __name__ == "__main__":
    r1 = grade_reasoning("Contains toxic insults", "Toxic insults")
    print(f"Good match: {r1}")
