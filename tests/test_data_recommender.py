import pytest
from app.data_recommender import recommend_datasets, extract_keywords

def test_extract_keywords():
    hypothesis = "Blue light enhances plant growth in high humidity environments"
    keywords = extract_keywords(hypothesis)
    assert isinstance(keywords, list)
    assert "light" in keywords
    assert "plant" in keywords
    assert "growth" in keywords
    assert "humidity" in keywords
    assert len(keywords) >= 4

def test_recommend_datasets():
    hypotheses = [
        "Blue light enhances plant growth",
        "Temperature affects protein folding"
    ]
    recommendations = recommend_datasets(hypotheses)
    assert isinstance(recommendations, list)
    assert len(recommendations) == len(hypotheses)
    for rec in recommendations:
        assert "hypothesis" in rec
        assert "datasets" in rec
        assert isinstance(rec["datasets"], list)
        assert len(rec["datasets"]) > 0

def test_recommend_datasets_empty_input():
    recommendations = recommend_datasets([])
    assert isinstance(recommendations, list)
    assert len(recommendations) == 0

def test_recommend_datasets_invalid_input():
    with pytest.raises(TypeError):
        recommend_datasets(None)  # type: ignore
    with pytest.raises(TypeError):
        recommend_datasets("not a list")  # type: ignore

def test_keyword_relevance():
    hypothesis = "Neural network depth affects model performance"
    keywords = extract_keywords(hypothesis)
    assert "neural" in keywords
    assert "network" in keywords
    assert "model" in keywords
    assert "performance" in keywords
    
    # Common words should be excluded
    common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with"}
    for word in common_words:
        assert word not in keywords 