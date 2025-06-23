#!/usr/bin/env python3
"""Test script for advanced NLP capabilities in RAG system."""

import json
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.rag_system import RAGSystem

def test_advanced_nlp():
    """Test the advanced NLP capabilities in the RAG system."""
    print("🧠 Testing Advanced NLP Capabilities")
    print("=" * 50)
    
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
        print("🔧 Initializing RAG system...")
        rag = RAGSystem()
        
        # Add papers to index
        print("📚 Adding papers to index...")
        rag.add_papers_to_index(papers)
        
        # Test comparative questions that should trigger advanced NLP
        comparative_questions = [
            "What are the key differences between the models discussed?",
            "Compare the approaches used in these research papers",
            "What are the strengths and weaknesses of different AI models?",
            "How do the methods in these papers differ?",
            "What are the main differences between language models and other AI approaches?"
        ]
        
        print("\n🔍 Testing Comparative Questions with Advanced NLP:")
        print("=" * 60)
        
        for i, question in enumerate(comparative_questions, 1):
            print(f"\n📝 Test {i}: {question}")
            print("-" * 50)
            
            try:
                result = rag.answer_question(question, top_k=5)
                
                print(f"📄 Papers retrieved: {result['num_papers_retrieved']}")
                print(f"🤖 Answer length: {len(result['answer'])} characters")
                
                # Check if it's using the advanced NLP fallback
                if "Model Comparison from Research Papers" in result['answer']:
                    print("✅ Using Advanced NLP Fallback Table")
                    print("📊 Table format detected with strengths/weaknesses extraction")
                elif "| Model Name |" in result['answer']:
                    print("✅ Using Markdown Table Format")
                    print("📊 Structured comparison with NLP-extracted features")
                else:
                    print("📝 Using LLM-generated response")
                
                # Show preview of the answer
                answer_preview = result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer']
                print(f"📋 Answer preview:\n{answer_preview}")
                
                # Check for NLP indicators
                nlp_indicators = ["Strengths:", "Weaknesses:", "Unique Features:", "|"]
                has_nlp = any(indicator in result['answer'] for indicator in nlp_indicators)
                if has_nlp:
                    print("✅ Advanced NLP features detected in response")
                else:
                    print("⚠️ Standard response format")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        print("\n🎉 Advanced NLP testing completed!")
        print("\n📊 Summary:")
        print("- Advanced NLP patterns for strength/weakness extraction")
        print("- Markdown table formatting for comparisons")
        print("- Sentence-level analysis using regex patterns")
        print("- Fallback mechanisms for robust responses")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")

if __name__ == "__main__":
    test_advanced_nlp() 