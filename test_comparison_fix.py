#!/usr/bin/env python3
"""Test script to verify improved comparison detection."""

import json
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.rag_system import RAGSystem


def test_comparison_detection():
    """Test the improved comparison detection and generic response handling."""
    print("🔍 Testing Improved Comparison Detection")
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
        rag = RAGSystem()
        rag.add_papers_to_index(papers)

        # Test the specific question that was failing
        test_question = "what is the difference between all these research papers ? and what different things I would learn ?"

        print(f"\n📝 Testing Question: {test_question}")
        print("-" * 60)

        # Test comparison detection
        comparison_keywords = [
            "difference",
            "compare",
            "contrast",
            "versus",
            "vs",
            "between",
            "models",
            "approaches",
            "methods",
            "different",
            "learn",
            "what different",
            "how do they differ",
            "distinguish",
            "vary",
            "variation",
        ]
        is_comparison = any(
            keyword in test_question.lower() for keyword in comparison_keywords
        )
        print(f"🔍 Comparison Detection: {is_comparison}")

        # Test generic response detection
        generic_response = "We should begin by attempting to define the subject matter."
        print(f"📝 Generic Response: {generic_response}")

        # Test the detection method
        sample_papers = rag.retrieve_relevant_papers("language models", top_k=3)
        is_generic = rag._is_repetitive_or_generic(generic_response, sample_papers)
        print(f"🔍 Generic Detection: {is_generic}")

        # Now test the full system
        print(f"\n🚀 Testing Full System Response:")
        print("-" * 40)

        result = rag.answer_question(test_question, top_k=5)

        print(f"📄 Papers retrieved: {result['num_papers_retrieved']}")
        print(f"🤖 Answer length: {len(result['answer'])} characters")

        # Check if it's using the advanced NLP fallback
        if "Model Comparison from Research Papers" in result["answer"]:
            print("✅ Using Advanced NLP Fallback Table")
        elif "| Model Name |" in result["answer"]:
            print("✅ Using Markdown Table Format")
        else:
            print("📝 Using LLM-generated response")

        # Show preview
        answer_preview = (
            result["answer"][:500] + "..."
            if len(result["answer"]) > 500
            else result["answer"]
        )
        print(f"📋 Answer preview:\n{answer_preview}")

        print("\n🎉 Test completed!")

    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")


if __name__ == "__main__":
    test_comparison_detection()
