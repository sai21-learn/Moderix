def grade_nsfw(content: str, predicted_category: str, gold_category: str) -> float:
    """
    Grade NSFW multi-class detection.
    
    Categories: "safe", "violence", "explicit", "adult_content"
    
    Args:
        content: Post text
        predicted_category: Agent's prediction
        gold_category: True label
    
    Returns:
        Reward (0.0-1.0)
    """
    if predicted_category == gold_category:
        # Perfect match
        return 1.0
    
    # Check if "close" (confusable categories)
    close_pairs = {
        ("violence", "explicit"),
        ("adult_content", "explicit"),
        ("safe", "violence"),
        ("safe", "explicit")
    }
    
    if (predicted_category, gold_category) in close_pairs or \
       (gold_category, predicted_category) in close_pairs:
        # One category off (understandable confusion)
        return 0.6
    else:
        # Completely wrong
        return 0.1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, default="")
    parser.add_argument("--predicted", type=str, default="safe")
    parser.add_argument("--gold", type=str, default="safe")
    args = parser.parse_args()

    if args.content:
        # Clinical invocation
        print(grade_nsfw(args.content, args.predicted, args.gold))
    else:
        # Internal test cases
        r1 = grade_nsfw("...", "violence", "violence")
        print(f"Perfect: {r1}")  # 1.0
        r2 = grade_nsfw("...", "explicit", "violence")
        print(f"One off: {r2}")  # 0.6
        r3 = grade_nsfw("...", "safe", "explicit")
        print(f"Wrong: {r3}")  # 0.1
