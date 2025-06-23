#!/usr/bin/env python3
"""Test script for RAG system improvements."""

import json
import sys
import os
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.rag_system import RAGSystem

def test_rag_improvements():
    """Test the improved RAG system."""
    print("🧪 Testing RAG System Improvements")
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
        
        # Test questions
        test_questions = [
            "What are the main findings about neural networks?",
            "What are the key insights about language models?",
            "What research hypotheses were generated?",
            "What are the main trends in AI research?",
            "What AI tools can I learn from these papers?",
            "Which research papers mention the use of blockchain for medical data privacy?",
            "Are there any studies on reinforcement learning for autonomous underwater vehicles in this collection?",
            "What are the main findings about CRISPR gene editing in these papers?",
            "Will these papers help in preparing an interview for AI engineer?",
            "What are the key differences between the models discussed?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🔍 Test {i}: {question}")
            print("-" * 40)
            
            try:
                result = rag.answer_question(question, top_k=3)
                
                print(f"📄 Papers retrieved: {result['num_papers_retrieved']}")
                print(f"🤖 Answer length: {len(result['answer'])} characters")
                print(f"📝 Answer preview: {result['answer'][:200]}...")
                
                if len(result['answer']) > 100:
                    print("✅ Generated substantial answer")
                else:
                    print("⚠️ Answer seems too short")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        print("\n🎉 RAG system test completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")

if __name__ == "__main__":
    test_rag_improvements() 