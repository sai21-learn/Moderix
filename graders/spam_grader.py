def grade_spam(content: str, predicted_is_spam: bool, gold_is_spam: bool) -> float:
    """
    Grade spam classification (binary).
    """
    if predicted_is_spam == gold_is_spam:
        reward = 1.0 if gold_is_spam else 0.9
    else:
        reward = 0.2 if (gold_is_spam and not predicted_is_spam) else 0.5
    return min(max(reward, 0.0), 1.0)


def grade(observation: dict, action: dict) -> float:
    """
    Standard OpenEnv grader interface for spam_classification task.

    Args:
        observation: dict with 'content_text' and 'spam' (bool gold label)
        action:      dict with 'decision', 'violation_type', 'confidence'

    Returns:
        Reward in [0.0, 1.0]
    """
    content    = observation.get("content_text", "")
    gold_spam  = bool(observation.get("spam", False))

    decision       = action.get("decision", "approve").lower()
    violation_type = action.get("violation_type", "none").lower()

    pred_spam = decision in ("reject", "ban_user") and violation_type in ("spam", "none")

    return grade_spam(content, pred_spam, gold_spam)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, default="")
    parser.add_argument("--predicted", type=str, default="false")
    parser.add_argument("--gold", type=str, default="false")
    args = parser.parse_args()

    if args.content:
        # Clinical invocation
        p = args.predicted.lower() == "true"
        g = args.gold.lower() == "true"
        print(grade_spam(args.content, p, g))
    else:
        # Internal test cases
        r1 = grade_spam("...", True, True)  # Caught spam
        print(f"Caught spam: {r1}")  # 1.0
        r2 = grade_spam("...", False, False)  # Safe post
        print(f"Safe post: {r2}")  # 0.9
        r3 = grade_spam("...", False, True)  # Missed spam
        print(f"Missed spam: {r3}")  # 0.2
