import pytest
from app.hypothesis_gen import generate_hypotheses
import re

def test_generate_hypotheses():
    # Test with sample insights
    insights = """Title:
    Neural Networks in AI
    Main Topic:
    This paper focuses on deep learning architectures
    Abstract Summary:
    Deep learning has revolutionized artificial intelligence.
    Key Points:
    1. Neural networks are powerful
    2. Training requires data
    3. Model architecture matters"""
    
    hypotheses = generate_hypotheses(insights)
    
    # Check basic structure
    assert isinstance(hypotheses, str)
    assert "Research Hypotheses:" in hypotheses
    
    # Check number of hypotheses
    hypotheses_lines = [line for line in hypotheses.split('\n') if line.strip().startswith(('1.', '2.', '3.'))]
    assert len(hypotheses_lines) >= 3
    
    # Check content relevance
    assert any('neural' in h.lower() for h in hypotheses_lines)
    assert any('learning' in h.lower() for h in hypotheses_lines)

def test_generate_hypotheses_empty_input():
    # Test with empty input
    hypotheses = generate_hypotheses("")
    assert isinstance(hypotheses, str)
    assert len(hypotheses.split('\n')) >= 2  # Should still generate some general hypotheses

def test_generate_hypotheses_minimal_input():
    # Test with minimal input
    insights = """Title:
    Test Paper
    Main Topic:
    Testing"""
    
    hypotheses = generate_hypotheses(insights)
    assert isinstance(hypotheses, str)
    assert "Research Hypotheses:" in hypotheses
    assert len(hypotheses.split('\n')) >= 2

def test_hypothesis_formatting():
    insights = """Title:
    Test Paper
    Main Topic:
    Testing methodology
    Key Points:
    1. First point
    2. Second point"""
    
    hypotheses = generate_hypotheses(insights)
    
    # Check formatting
    lines = hypotheses.split('\n')
    assert lines[0] == "Research Hypotheses:"
    
    # Check numbering
    numbered_lines = [l for l in lines if l.strip() and l[0].isdigit()]
    assert len(numbered_lines) >= 2  # Should have at least 2 hypotheses
    
    # Check each line starts with a number and period
    for line in numbered_lines:
        assert re.match(r'^\d+\.', line) 