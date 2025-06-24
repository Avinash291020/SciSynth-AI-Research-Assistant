import nbformat as nbf

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
    "source": """# SciSynth Visualization Examples 📊

This notebook demonstrates various visualization techniques available in SciSynth for analyzing research results and hypotheses.""",
}
nb.cells.append(nbf.from_dict(cell0))

# Cell 1 - Code
cell1 = {
    "cell_type": "code",
    "metadata": {},
    "source": """import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
plt.style.use('seaborn')
sns.set_palette("husl")""",
    "execution_count": None,
    "outputs": [],
}
nb.cells.append(nbf.from_dict(cell1))

# Cell 2 - Markdown
cell2 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": """## 1. Hypothesis Confidence Distribution

First, let's visualize the confidence scores for different hypotheses:""",
}
nb.cells.append(nbf.from_dict(cell2))

# Cell 3 - Code
cell3 = {
    "cell_type": "code",
    "metadata": {},
    "source": """# Sample data
hypotheses = [
    "H1: Light intensity affects growth",
    "H2: Temperature modulates protein folding",
    "H3: pH influences enzyme activity",
    "H4: Substrate concentration impacts rate",
    "H5: Time affects yield"
]
confidence_scores = [0.92, 0.85, 0.78, 0.88, 0.71]

# Create bar plot
plt.figure(figsize=(12, 6))
bars = plt.bar(hypotheses, confidence_scores, color=sns.color_palette("husl", 5))
plt.xticks(rotation=45, ha='right')
plt.title('Hypothesis Confidence Scores')
plt.ylabel('Confidence Score')
plt.ylim(0, 1)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()""",
    "execution_count": None,
    "outputs": [],
}
nb.cells.append(nbf.from_dict(cell3))

# Cell 4 - Markdown
cell4 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": """## 2. Research Progress Timeline

Next, let's create a timeline visualization of research milestones:""",
}
nb.cells.append(nbf.from_dict(cell4))

# Cell 5 - Code
cell5 = {
    "cell_type": "code",
    "metadata": {},
    "source": """# Sample timeline data
milestones = [
    'Literature Review',
    'Hypothesis Generation',
    'Dataset Collection',
    'Initial Testing',
    'Model Development',
    'Results Analysis'
]
start_dates = pd.date_range('2024-01-01', periods=6, freq='W')
durations = [14, 7, 10, 5, 12, 8]  # days

# Create timeline DataFrame
df = pd.DataFrame({
    'Task': milestones,
    'Start': start_dates,
    'Duration': durations
})
df['End'] = df['Start'] + pd.to_timedelta(df['Duration'], unit='D')

# Plot timeline
fig, ax = plt.subplots(figsize=(12, 6))

# Plot horizontal bars
for idx, row in df.iterrows():
    ax.barh(row['Task'], 
            (row['End'] - row['Start']).days,
            left=row['Start'],
            color=sns.color_palette("husl", 6)[idx],
            alpha=0.8)

# Customize plot
ax.set_title('Research Progress Timeline')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()""",
    "execution_count": None,
    "outputs": [],
}
nb.cells.append(nbf.from_dict(cell5))

# Cell 6 - Markdown
cell6 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": """## 3. Model Performance Comparison

Finally, let's create a heatmap to compare different model performances:""",
}
nb.cells.append(nbf.from_dict(cell6))

# Cell 7 - Code
cell7 = {
    "cell_type": "code",
    "metadata": {},
    "source": """# Sample model performance data
models = ['Linear Regression', 'Random Forest', 'Neural Network', 'XGBoost']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# Generate random performance data
np.random.seed(42)
performance_data = np.random.uniform(0.7, 0.95, size=(len(models), len(metrics)))

# Create heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(performance_data, 
            annot=True, 
            fmt='.2f',
            xticklabels=metrics,
            yticklabels=models,
            cmap='YlOrRd',
            vmin=0.7,
            vmax=1.0)

plt.title('Model Performance Comparison')
plt.tight_layout()
plt.show()""",
    "execution_count": None,
    "outputs": [],
}
nb.cells.append(nbf.from_dict(cell7))

# Write the notebook to a file
with open("examples/visualization_examples.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
