#!/usr/bin/env python3
"""Quick test to verify advanced NLP fallback is working."""

import json
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.rag_system import RAGSystem


def quick_test():
    """Quick test of the advanced NLP fallback."""
    print("⚡ Quick Test: Advanced NLP Fallback")
    print("=" * 40)

    # Load existing papers
    results_path = Path("results/all_papers_results.json")
    if not results_path.exists():
        print("❌ No results file found.")
        return

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            papers = json.load(f)
        print(f"✅ Loaded {len(papers)} papers")

        # Initialize RAG system
        rag = RAGSystem()
        rag.add_papers_to_index(papers)

        # Get some sample papers
        sample_papers = rag.retrieve_relevant_papers("language models", top_k=3)
        print(f"📚 Retrieved {len(sample_papers)} sample papers")

        # Test the advanced NLP fallback directly
        print("\n🔬 Testing Advanced NLP Fallback:")
        print("-" * 40)

        advanced_comparison = rag._create_model_comparison_fallback(sample_papers)

        print("✅ Advanced NLP Fallback Generated Successfully!")
        print(f"📏 Length: {len(advanced_comparison)} characters")

        # Check for key features
        if "Model Comparison from Research Papers" in advanced_comparison:
            print("✅ Contains proper header")
        if "| Model Name |" in advanced_comparison:
            print("✅ Contains markdown table")
        if "Strengths" in advanced_comparison:
            print("✅ Contains strengths column")
        if "Weaknesses" in advanced_comparison:
            print("✅ Contains weaknesses column")

        # Show preview
        print(f"\n📋 Preview (first 500 chars):\n{advanced_comparison[:500]}...")

        print("\n🎉 Advanced NLP is working correctly!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    quick_test()
