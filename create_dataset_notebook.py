import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}

# Cell 0 - Markdown
cell0 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": """# Dataset Recommendation Example 📊

This notebook demonstrates how to use SciSynth's dataset recommendation system to find relevant datasets for research hypotheses. The system analyzes hypotheses and suggests datasets that could be useful for testing them.

## Setup
First, let's import the required modules:"""
}
nb.cells.append(nbf.from_dict(cell0))

# Cell 1 - Code
cell1 = {
    "cell_type": "code",
    "metadata": {},
    "source": """from app.data_recommender import recommend_datasets, extract_keywords
import pandas as pd""",
    "execution_count": None,
    "outputs": []
}
nb.cells.append(nbf.from_dict(cell1))

# Cell 2 - Markdown
cell2 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": """## 1. Extract Keywords from Hypotheses

Let's start by extracting keywords from some example hypotheses:"""
}
nb.cells.append(nbf.from_dict(cell2))

# Cell 3 - Code
cell3 = {
    "cell_type": "code",
    "metadata": {},
    "source": """hypothesis = "Neural networks perform better with larger training datasets"
keywords = extract_keywords(hypothesis)
print("Extracted Keywords:")
print(keywords)""",
    "execution_count": None,
    "outputs": []
}
nb.cells.append(nbf.from_dict(cell3))

# Cell 4 - Markdown
cell4 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": """## 2. Get Dataset Recommendations

Now let's get dataset recommendations for multiple hypotheses:"""
}
nb.cells.append(nbf.from_dict(cell4))

# Cell 5 - Code
cell5 = {
    "cell_type": "code",
    "metadata": {},
    "source": """hypotheses = [
    "Neural networks perform better with larger training datasets",
    "Transformer models excel at natural language understanding tasks",
    "Deep learning models require significant computational resources"
]

recommendations = recommend_datasets(hypotheses)

print("Dataset Recommendations:")
for rec in recommendations:
    print(f"\\nHypothesis: {rec['hypothesis']}")
    print("Recommended Datasets:")
    for dataset in rec['datasets']:
        print(f"- {dataset}")""",
    "execution_count": None,
    "outputs": []
}
nb.cells.append(nbf.from_dict(cell5))

# Cell 6 - Markdown
cell6 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": """## 3. Analyze Recommendations

Let's create a summary of the recommendations:"""
}
nb.cells.append(nbf.from_dict(cell6))

# Cell 7 - Code
cell7 = {
    "cell_type": "code",
    "metadata": {},
    "source": """# Create a summary DataFrame
summary_data = []
for rec in recommendations:
    summary_data.append({
        'hypothesis': rec['hypothesis'],
        'num_datasets': len(rec['datasets']),
        'top_dataset': rec['datasets'][0] if rec['datasets'] else 'None'
    })

summary_df = pd.DataFrame(summary_data)
print("\\nRecommendation Summary:")
print(summary_df)""",
    "execution_count": None,
    "outputs": []
}
nb.cells.append(nbf.from_dict(cell7))

# Write the notebook to a file
with open('examples/dataset_recommendation.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 