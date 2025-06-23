def check_consistency(hypothesis: str) -> bool:
    if any(word in hypothesis.lower() for word in ["not", "no", "null", "none"]):
        return False
    return True  # Simplified rule

if __name__ == "__main__":
    tests = [
        "X increases Y when Z is present.",
        "There is no relationship between A and B."
    ]
    for t in tests:
        print(f"{t} → {'Valid' if check_consistency(t) else 'Contradictory'}") 