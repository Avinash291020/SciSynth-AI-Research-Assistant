#!/usr/bin/env python3
"""Demonstration of Advanced NLP capabilities in RAG system."""

import json
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.rag_system import RAGSystem

def demo_advanced_nlp():
    """Demonstrate the advanced NLP capabilities."""
    print("🧠 Advanced NLP Demonstration")
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
        rag = RAGSystem()
        
        # Add papers to index
        rag.add_papers_to_index(papers)
        
        # Get some sample papers for demonstration
        sample_papers = rag.retrieve_relevant_papers("language models neural networks", top_k=5)
        
        print(f"\n📚 Retrieved {len(sample_papers)} sample papers for demonstration")
        
        # Demonstrate the advanced NLP fallback
        print("\n🔬 Advanced NLP Model Comparison Fallback:")
        print("=" * 60)
        
        # Call the advanced NLP method directly
        advanced_comparison = rag._create_model_comparison_fallback(sample_papers)
        
        print(advanced_comparison)
        
        print("\n🎯 Key Advanced NLP Features Demonstrated:")
        print("=" * 50)
        print("✅ Pattern-based strength/weakness extraction")
        print("✅ Sentence-level analysis using regex")
        print("✅ Markdown table formatting")
        print("✅ Model name extraction from titles")
        print("✅ Year extraction from metadata")
        print("✅ Technical terms integration")
        print("✅ Fallback mechanisms for robustness")
        
        print("\n📊 NLP Patterns Used:")
        print("- Strength patterns: improve, outperform, robust, advantage, effective, etc.")
        print("- Weakness patterns: limitation, fail, struggle, drawback, require more, etc.")
        print("- Sentence segmentation and analysis")
        print("- Contextual feature extraction")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")

if __name__ == "__main__":
    demo_advanced_nlp() 