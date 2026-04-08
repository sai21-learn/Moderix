def grade_toxicity(content: str, predicted_score: float, gold_label_score: float) -> float:
    """
    Grade toxicity detection.
    
    Args:
        content: Post text (for reference)
        predicted_score: Agent's toxicity score (0.0-1.0)
        gold_label_score: True label (0.0-1.0)
    
    Returns:
        Reward (0.0-1.0)
    """
    # Sigmoid accuracy: penalize distance from gold label
    distance = abs(predicted_score - gold_label_score)
    accuracy = max(0.0, 1.0 - distance)
    
    confidence_bonus = 0.0
    if gold_label_score > 0.5 and predicted_score > 0.5:
        confidence_bonus = 0.1 * min(predicted_score, gold_label_score)
    elif gold_label_score < 0.5 and predicted_score < 0.5:
        confidence_bonus = 0.1 * (1 - max(predicted_score, gold_label_score))
    
    reward = accuracy + confidence_bonus * 0.1
    return min(max(reward, 0.0), 1.0)


def grade(observation: dict, action: dict) -> float:
    """
    Standard OpenEnv grader interface for toxicity_detection task.

    Args:
        observation: dict with at least 'content_text' and 'toxicity' (gold label)
        action:      dict with 'decision', 'violation_type', 'confidence'

    Returns:
        Reward in [0.0, 1.0]
    """
    content    = observation.get("content_text", "")
    gold_score = float(observation.get("toxicity", 0.0))

    decision       = action.get("decision", "approve").lower()
    violation_type = action.get("violation_type", "none").lower()
    confidence     = float(action.get("confidence", 0.5))

    # Map decision → predicted toxicity score
    if decision in ("reject", "ban_user") and violation_type in ("toxicity", "none"):
        pred_score = 0.9 * confidence
    elif decision in ("review", "escalate"):
        pred_score = 0.5
    else:
        pred_score = 0.1

    return grade_toxicity(content, pred_score, gold_score)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, default="")
    parser.add_argument("--predicted", type=float, default=0.0)
    parser.add_argument("--gold", type=float, default=0.0)
    args = parser.parse_args()

    if args.content:
        # Clinical invocation
        print(grade_toxicity(args.content, args.predicted, args.gold))
    else:
        # Internal test cases
        r1 = grade_toxicity("bad content", 0.9, 0.9)
        print(f"Perfect: {r1}")
        r2 = grade_toxicity("bad content", 0.7, 0.9)
        print(f"Off by 0.2: {r2}")
        r3 = grade_toxicity("bad content", 0.1, 0.9)
        print(f"Wrong: {r3}")
