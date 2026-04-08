def grade_spam(content: str, predicted_is_spam: bool, gold_is_spam: bool) -> float:
    """
    Grade spam classification (binary).
    
    Args:
        content: Post text (for reference)
        predicted_is_spam: Agent's binary prediction
        gold_is_spam: True label
    
    Returns:
        Reward (0.0-1.0)
    """
    # Basic accuracy
    if predicted_is_spam == gold_is_spam:
        # Correct prediction
        if gold_is_spam:
            # Caught spam (important, recall matters)
            reward = 1.0
        else:
            # Correctly approved safe post
            reward = 0.9
    else:
        # Incorrect prediction
        if gold_is_spam and not predicted_is_spam:
            # Missed spam (worse than false positive)
            reward = 0.2
        else:
            # False positive (less bad)
            reward = 0.5
    
    return min(max(reward, 0.0), 1.0)

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
