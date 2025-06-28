# -*- coding: utf-8 -*-
"""
Setup configuration for SciSynth AI Research Assistant.
"""

from setuptools import setup, find_packages

setup(
    name="scisynth",
    version="0.1.0",
    description="An Autonomous AI Research Assistant",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Avinash Kumar",
    author_email="ak3578431@gmail.com",
    packages=find_packages(),
    install_requires=[
        "PyMuPDF",  # PDF processing library
        "llama-index",
        "langchain",
        "langchain-community",
        "openai",
        "streamlit",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "sympy",
        "pytest",
        "fastapi",
        "python-multipart",
        "uvicorn",
        "langgraph",
        "python-dotenv",
        "PyPDF2",
        "sentence-transformers",  # For local embeddings
        "transformers",  # For local LLM support
        "torch",  # Required for transformers
        "networkx",  # For citation network analysis
        "tqdm",  # For progress bars
        "numpy",  # Required for various computations
        "plotly",  # For interactive visualizations
        "seaborn",  # For statistical visualizations
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
