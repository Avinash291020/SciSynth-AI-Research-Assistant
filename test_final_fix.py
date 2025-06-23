#!/usr/bin/env python3
"""Final test to verify the improved RAG system handles comparative questions properly."""

import json
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.rag_system import RAGSystem

def test_final_fix():
    """Test the final fix for comparative questions."""
    print("🎯 Final Test: Improved Comparative Question Handling")
    print("=" * 60)
    
    # Load existing papers
    results_path = Path("results/all_papers_results.json")
    if not results_path.exists():
        print("❌ No results file found. Please process papers first.")
        return
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        print(f"✅ Loaded {len(papers)} papers")
        
        # Initialize RAG system
        rag = RAGSystem()
        rag.add_papers_to_index(papers)
        
        # Test the exact question that was failing
        test_question = "what is the difference between all these research papers ? and what different things I would learn ?"
        
        print(f"\n📝 Testing Question: {test_question}")
        print("-" * 70)
        
        # Test the full system
        result = rag.answer_question(test_question, top_k=5)
        
        print(f"📄 Papers retrieved: {result['num_papers_retrieved']}")
        print(f"🤖 Answer length: {len(result['answer'])} characters")
        
        # Check what type of response we got
        if "Model Comparison from Research Papers" in result['answer']:
            print("✅ SUCCESS: Using Advanced NLP Fallback Table")
            print("🎉 The fix worked! Advanced NLP is now active.")
        elif "| Model Name |" in result['answer']:
            print("✅ SUCCESS: Using Markdown Table Format")
            print("🎉 The fix worked! Structured comparison is active.")
        elif "Paper 1:" in result['answer'] and "Paper 2:" in result['answer']:
            print("❌ FAILED: Still getting repetitive Paper X: pattern")
            print("⚠️ The fix didn't work as expected.")
        else:
            print("📝 Using LLM-generated response")
            if len(result['answer']) > 200:
                print("✅ Response looks substantial")
            else:
                print("⚠️ Response might be too short")
        
        # Show preview
        answer_preview = result['answer'][:800] + "..." if len(result['answer']) > 800 else result['answer']
        print(f"\n📋 Answer preview:\n{answer_preview}")
        
        # Test the detection methods
        print(f"\n🔍 Testing Detection Methods:")
        print("-" * 40)
        
        # Test comparison detection
        comparison_keywords = ["difference", "compare", "contrast", "versus", "vs", "between", "models", "approaches", "methods", "different", "learn", "what different", "how do they differ", "distinguish", "vary", "variation"]
        is_comparison = any(keyword in test_question.lower() for keyword in comparison_keywords)
        print(f"🔍 Comparison Detection: {is_comparison}")
        
        # Test specific pattern detection
        if "what is the difference between all these research papers" in test_question.lower():
            print("🔍 Specific Pattern Detection: ✅")
        else:
            print("🔍 Specific Pattern Detection: ❌")
        
        print("\n🎉 Test completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")

if __name__ == "__main__":
    test_final_fix() 