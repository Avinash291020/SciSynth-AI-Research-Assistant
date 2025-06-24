import nbformat as nbf
from jupyter_client.kernelspec import KernelSpecManager

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

# Cell 0 - Markdown
cell0 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": """# SciSynth: Basic Research Pipeline Example 🧪

This notebook demonstrates the basic workflow of using SciSynth to analyze a research paper and generate insights. The pipeline includes:

1. Loading and processing a research paper
2. Generating insights from the paper
3. Generating hypotheses based on the insights
4. Testing a simple model on sample data
5. Visualizing results and citation networks

## Setup
First, let's import the required modules:""",
}
nb.cells.append(nbf.from_dict(cell0))

# Cell 1 - Code
cell1 = {
    "cell_type": "code",
    "metadata": {},
    "source": """import os
from pathlib import Path
from app.ingest_paper import extract_text_from_pdf, chunk_and_index
from app.insight_agent import generate_insights
from app.hypothesis_gen import generate_hypotheses
from app.model_tester import run_basic_model
from app.citation_network import CitationNetwork
import matplotlib.pyplot as plt
import pandas as pd""",
    "execution_count": None,
    "outputs": [],
}
nb.cells.append(nbf.from_dict(cell1))

# Cell 2 - Markdown
cell2 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": """## 1. Load and Process a Research Paper

First, we'll load a sample research paper and process it into manageable chunks:""",
}
nb.cells.append(nbf.from_dict(cell2))

# Cell 3 - Code
cell3 = {
    "cell_type": "code",
    "metadata": {},
    "source": """# Example with a sample paper
paper_path = "data/sample_paper.pdf"
text = extract_text_from_pdf(paper_path)
index = chunk_and_index(text)
print(f"Processed {len(text.split())} words from the paper")""",
    "execution_count": None,
    "outputs": [],
}
nb.cells.append(nbf.from_dict(cell3))

# Cell 4 - Markdown
cell4 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": """## 2. Generate Insights

Next, we'll use our insight generation system to extract key points from the paper:""",
}
nb.cells.append(nbf.from_dict(cell4))

# Cell 5 - Code
cell5 = {
    "cell_type": "code",
    "metadata": {},
    "source": """insights = generate_insights(index)
print("Key Insights:")
for i, insight in enumerate(insights.split('\\n'), 1):
    if insight.strip():
        print(f"{i}. {insight.strip()}")""",
    "execution_count": None,
    "outputs": [],
}
nb.cells.append(nbf.from_dict(cell5))

# Cell 6 - Markdown
cell6 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": """## 3. Generate Hypotheses

Based on the insights, we'll generate testable research hypotheses:""",
}
nb.cells.append(nbf.from_dict(cell6))

# Cell 7 - Code
cell7 = {
    "cell_type": "code",
    "metadata": {},
    "source": """hypotheses = generate_hypotheses(insights)
print("Generated Hypotheses:")
for i, hypothesis in enumerate(hypotheses.split('\\n'), 1):
    if hypothesis.strip():
        print(f"{i}. {hypothesis.strip()}")""",
    "execution_count": None,
    "outputs": [],
}
nb.cells.append(nbf.from_dict(cell7))

# Cell 8 - Markdown
cell8 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": """## 4. Test a Simple Model

Let's test a simple model on our sample dataset to evaluate our hypotheses:""",
}
nb.cells.append(nbf.from_dict(cell8))

# Cell 9 - Code
cell9 = {
    "cell_type": "code",
    "metadata": {},
    "source": """# Example with a sample dataset
dataset_path = "data/sample_dataset.csv"
results = run_basic_model(dataset_path, target_column="growth_rate")
print("Model Results:")
for metric, value in results.items():
    if metric == "feature_importance":
        print("\\nFeature Importance:")
        for feature, importance in value.items():
            print(f"{feature}: {importance:.3f}")
    else:
        print(f"\\n{metric}: {value:.3f}")""",
    "execution_count": None,
    "outputs": [],
}
nb.cells.append(nbf.from_dict(cell9))

# Cell 10 - Markdown
cell10 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": """## 5. Visualize Results

Finally, let's create visualizations of our results and the citation network:""",
}
nb.cells.append(nbf.from_dict(cell10))

# Cell 11 - Code
cell11 = {
    "cell_type": "code",
    "metadata": {},
    "source": """# Visualize feature importance
plt.figure(figsize=(10, 6))
features = list(results['feature_importance'].keys())
importances = list(results['feature_importance'].values())

plt.bar(features, importances)
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

# Visualize citation network
network = CitationNetwork()
graph, relationships = network.analyze_papers(Path("results"))

plt.figure(figsize=(12, 8))
network.visualize_network()
plt.show()""",
    "execution_count": None,
    "outputs": [],
}
nb.cells.append(nbf.from_dict(cell11))

# Write the notebook to a file
with open("examples/basic_research_pipeline.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
