import pytest
from app.insight_agent import generate_insights, clean_title, extract_title_and_abstract, extract_topic
from app.ingest_paper import chunk_and_index

def test_generate_insights_with_valid_input(sample_text):
    index = chunk_and_index(sample_text)
    insights = generate_insights(index)
    assert isinstance(insights, str)
    assert "Title:" in insights
    assert "Main Topic:" in insights
    assert "Key Points:" in insights

def test_generate_insights_with_empty_input():
    index = chunk_and_index("")
    insights = generate_insights(index)
    assert isinstance(insights, str)
    assert insights == "No text content available for analysis."

def test_generate_insights_preserves_scientific_terms(sample_text):
    index = chunk_and_index(sample_text)
    insights = generate_insights(index)
    # Check for presence of key terms from the text
    assert "blue light" in insights.lower()
    assert "growth rates" in insights.lower()

def test_generate_insights_formatting():
    text = "Test hypothesis about growth. Another test about light."
    index = chunk_and_index(text)
    insights = generate_insights(index)
    assert isinstance(insights, str)
    lines = [line for line in insights.split('\n') if line.strip()]
    # Check section headers
    assert any(line.strip() == "Title:" for line in lines)
    assert any(line.strip() == "Main Topic:" for line in lines)
    assert any(line.strip() == "Key Points:" for line in lines)

def test_clean_title():
    # Test basic cleaning
    title = "Deep Learning in Practice (2023)"
    assert clean_title(title) == "Deep Learning in Practice"
    
    # Test with author names
    title = "Neural Networks, Smith et al."
    assert clean_title(title) == "Neural Networks"
    
    # Test with special characters
    title = "AI & Machine Learning: A Review"
    assert clean_title(title) == "AI Machine Learning: A Review"  # Colons are preserved

def test_extract_title_and_abstract():
    text = """Deep Learning in Practice
    
    Abstract
    This is a test abstract about deep learning.
    It contains multiple lines.
    
    Keywords: AI, ML
    Introduction
    The rest of the paper..."""
    
    title, abstract = extract_title_and_abstract(text)
    assert title == "Deep Learning in Practice"
    assert "test abstract" in abstract
    assert "Keywords" not in abstract
    assert "Introduction" not in abstract

def test_extract_topic():
    title = "Deep Learning: A Comprehensive Review"
    abstract = "This paper reviews deep learning methods."
    topic = extract_topic(title, abstract)
    
    assert "Deep Learning" in topic
    assert len(topic.split()) <= 10  # Should not be too long

def test_generate_insights():
    chunks = [
        """Deep Learning Applications
        
        Abstract
        This paper explores deep learning.
        
        Introduction
        Deep learning has transformed AI.""",
        "Neural networks are powerful tools.",
        "Training requires significant data."
    ]
    
    insights = generate_insights(chunks)
    
    # Check structure
    assert "Title:" in insights
    assert "Main Topic:" in insights
    assert "Abstract Summary:" in insights
    assert "Key Points:" in insights
    
    # Check content
    assert "deep learning" in insights.lower()
    assert "applications" in insights.lower()

def test_generate_insights_minimal_input():
    chunks = ["Short paper about AI."]
    insights = generate_insights(chunks)
    
    assert isinstance(insights, str)
    assert "Title:" in insights
    assert "Main Topic:" in insights 