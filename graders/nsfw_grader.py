def grade_nsfw(content: str, predicted_category: str, gold_category: str) -> float:
    """
    Grade NSFW multi-class detection.
    Categories: "safe", "violence", "explicit", "adult_content"
    """
    if predicted_category == gold_category:
        return 1.0
    close_pairs = {
        ("violence", "explicit"),
        ("adult_content", "explicit"),
        ("safe", "violence"),
        ("safe", "explicit")
    }
    if (predicted_category, gold_category) in close_pairs or \
       (gold_category, predicted_category) in close_pairs:
        return 0.6
    return 0.1


def grade(observation: dict, action: dict) -> float:
    """
    Standard OpenEnv grader interface for nsfw_detection task.

    Args:
        observation: dict with 'content_text' and 'nsfw_category' (gold label)
        action:      dict with 'decision', 'violation_type', 'confidence'

    Returns:
        Reward in [0.0, 1.0]
    """
    content       = observation.get("content_text", "")
    gold_category = observation.get("nsfw_category", "safe")

    decision       = action.get("decision", "approve").lower()
    violation_type = action.get("violation_type", "none").lower()

    # Map violation_type to NSFW category
    nsfw_map = {
        "violence":       "violence",
        "explicit":       "explicit",
        "adult_content":  "adult_content",
        "nsfw":           gold_category,  # agent said nsfw but didn't specify - give benefit
    }
    if decision in ("reject", "ban_user"):
        pred_category = nsfw_map.get(violation_type, "safe")
    else:
        pred_category = "safe"

    return grade_nsfw(content, pred_category, gold_category)

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
