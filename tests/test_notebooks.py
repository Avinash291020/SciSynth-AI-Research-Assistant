import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
from pathlib import Path

def test_notebook_structure():
    """Test that all notebooks have the expected structure."""
    notebook_dir = Path("examples")
    notebooks = list(notebook_dir.glob("*.ipynb"))
    assert len(notebooks) > 0, "No notebooks found in examples directory"
    
    for nb_path in notebooks:
        with open(nb_path, encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            
        # Check basic structure
        assert len(nb.cells) > 0, f"Notebook {nb_path} has no cells"
        
        # Check for markdown documentation
        has_markdown = any(cell.cell_type == "markdown" for cell in nb.cells)
        assert has_markdown, f"Notebook {nb_path} has no markdown documentation"
        
        # Check for code cells
        has_code = any(cell.cell_type == "code" for cell in nb.cells)
        assert has_code, f"Notebook {nb_path} has no code cells"
        
        # Check for imports in first code cell
        first_code_cell = next(cell for cell in nb.cells if cell.cell_type == "code")
        assert "import" in first_code_cell.source.lower(), f"Notebook {nb_path} should have imports in first code cell"

def test_basic_research_pipeline():
    """Test the basic research pipeline notebook."""
    nb_path = "examples/basic_research_pipeline.ipynb"
    with open(nb_path, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Check for required components
    sources = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    all_code = "\n".join(sources)
    
    # Check for key functionality
    assert "generate_insights" in all_code, "Missing insight generation"
    assert "generate_hypotheses" in all_code, "Missing hypothesis generation"
    assert "print" in all_code, "Missing output display"

def test_visualization_examples():
    """Test the visualization examples notebook."""
    nb_path = "examples/visualization_examples.ipynb"
    with open(nb_path, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Check for visualization imports
    sources = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    all_code = "\n".join(sources)
    
    assert any("matplotlib" in src or "plt" in src for src in sources), "Missing matplotlib"
    assert "show()" in all_code or "savefig" in all_code, "Missing plot display/save"

def test_dataset_recommendation():
    """Test the dataset recommendation notebook."""
    nb_path = "examples/dataset_recommendation.ipynb"
    with open(nb_path, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Check for required components
    sources = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    all_code = "\n".join(sources)
    
    assert "recommend_datasets" in all_code, "Missing dataset recommendation"
    assert "extract_keywords" in all_code, "Missing keyword extraction"

@pytest.mark.skip(reason="Notebook execution tests are slow and require all dependencies")
def test_notebook_execution():
    """Test that notebooks can execute without errors."""
    notebook_dir = Path("examples")
    notebooks = list(notebook_dir.glob("*.ipynb"))
    
    for nb_path in notebooks:
        with open(nb_path, encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        try:
            ep.preprocess(nb, {'metadata': {'path': 'examples/'}})
        except Exception as e:
            pytest.fail(f"Error executing {nb_path}: {str(e)}") 