import re

def extract_keywords(hypothesis: str) -> list[str]:
    """Extract relevant keywords from a hypothesis string."""
    words = re.findall(r'\b\w+\b', hypothesis.lower())
    return [word for word in words if len(word) > 4]

def recommend_datasets(hypotheses) -> list[dict]:
    """Recommend datasets based on hypotheses keywords."""
    if not isinstance(hypotheses, list):
        raise TypeError("Input must be a list of hypotheses")
        
    recommendations = []
    for h in hypotheses:
        keywords = extract_keywords(h)
        # Mock dataset suggestions based on keywords
        datasets = [f"{kw}_dataset.csv" for kw in keywords[:2]]
        recommendations.append({
            "hypothesis": h,
            "datasets": datasets
        })
    return recommendations 