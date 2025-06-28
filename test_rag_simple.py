# -*- coding: utf-8 -*-
"""
Test suite for simple RAG functionality.
"""

#!/usr/bin/env python3
"""Simple test for improved RAG system."""

import json
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.rag_system import RAGSystem


def test_comparison():
    """Test the comparison functionality."""
    print("🧪 Testing RAG System Comparison")
    print("=" * 50)

    # Load existing papers
    results_path = Path("results/all_papers_results.json")
    if not results_path.exists():
        print("❌ No results file found. Please process papers first.")
        return

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            papers = json.load(f)
        print(f"✅ Loaded {len(papers)} papers")

        # Initialize RAG system
        print("🔧 Initializing RAG system...")
        rag = RAGSystem()

        # Add papers to index
        print("📚 Adding papers to index...")
        rag.add_papers_to_index(papers)

        # Test the comparison question
        question = "what is the difference between these papers?"
        print(f"\n🔍 Testing: {question}")
        print("-" * 40)

        result = rag.answer_question(question, top_k=4)

        print(f"📄 Papers retrieved: {result['num_papers_retrieved']}")
        print(f"🤖 Answer length: {len(result['answer'])} characters")
        print(f"📝 Answer:")
        print(result["answer"])

        print("\n🎉 Test completed!")

    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_comparison()
