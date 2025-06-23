import pytest
from pathlib import Path
import json
import networkx as nx
from app.citation_network import CitationNetwork

@pytest.fixture
def sample_results(tmp_path):
    """Create sample paper results for testing."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Create two sample paper results
    paper1 = {
        "paper_name": "paper1.pdf",
        "processed_date": "2024-01-01T00:00:00",
        "insights": "Title:\nNeural Networks in AI\nMain Topic:\nThis paper focuses on deep learning architectures\nKey Points:\n1. Deep learning has revolutionized AI\n2. Training requires significant data",
        "hypotheses": "Research Hypotheses:\n1. Based on Smith et al. 2023, neural networks will continue to evolve."
    }
    
    paper2 = {
        "paper_name": "paper2.pdf",
        "processed_date": "2024-01-02T00:00:00",
        "insights": "Title:\nAdvances in Deep Learning\nMain Topic:\nThis paper focuses on neural network architectures\nKey Points:\n1. Modern architectures are becoming more efficient\n2. Citing [1] shows the progress",
        "hypotheses": "Research Hypotheses:\n1. Building on (Johnson, 2023), efficiency will improve."
    }
    
    # Save sample results
    for paper in [paper1, paper2]:
        with open(results_dir / f"{paper['paper_name']}_results.json", 'w') as f:
            json.dump(paper, f)
            
    return results_dir

def test_extract_citations():
    network = CitationNetwork()
    text = """
    Recent work [1, 2] has shown progress.
    Smith et al. 2023 demonstrated that...
    According to (Johnson, 2023), the method...
    """
    citations = network.extract_citations(text)
    assert len(citations) >= 3
    assert "1, 2" in citations
    assert "Smith et al. 2023" in citations
    assert "Johnson, 2023" in citations

def test_compute_similarity():
    network = CitationNetwork()
    text1 = "Deep learning has revolutionized artificial intelligence"
    text2 = "Neural networks have transformed AI applications"
    similarity = network.compute_similarity(text1, text2)
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1
    
    # Same text should have high similarity
    same_similarity = network.compute_similarity(text1, text1)
    assert same_similarity > 0.9

def test_analyze_papers(sample_results):
    network = CitationNetwork()
    graph, relationships = network.analyze_papers(sample_results)
    
    # Check graph properties
    assert isinstance(graph, nx.DiGraph)
    assert len(list(graph.nodes())) >= 2  # At least the two papers
    assert len(list(graph.edges())) >= 1  # At least one citation
    
    # Check relationships
    assert isinstance(relationships, list)
    assert len(relationships) >= 1  # Papers should be related (similar topics)
    for rel in relationships:
        assert "paper1" in rel
        assert "paper2" in rel
        assert "similarity" in rel
        assert "shared_topics" in rel

def test_find_shared_topics():
    network = CitationNetwork()
    insights1 = """Title:
    Neural Networks
    Key Points:
    1. Deep learning is powerful
    2. Training requires data"""
    
    insights2 = """Title:
    Deep Learning
    Key Points:
    1. Neural networks are effective
    2. Data is essential"""
    
    shared = network._find_shared_topics(insights1, insights2)
    assert isinstance(shared, list)
    assert len(shared) >= 1  # Should find some shared topics

def test_visualization(sample_results, tmp_path):
    network = CitationNetwork()
    graph, _ = network.analyze_papers(sample_results)
    
    output_path = tmp_path / "test_network.png"
    network.visualize_network(str(output_path))
    assert output_path.exists()
    assert output_path.stat().st_size > 0  # File should not be empty 