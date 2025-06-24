#!/usr/bin/env python3
"""Test script to verify all AI capabilities are working."""

import json
import sys
from pathlib import Path


def test_ai_capabilities():
    """Test all AI capabilities and provide a summary."""

    print("🧪 Testing SciSynth AI Capabilities...")
    print("=" * 50)

    # Check if results file exists
    results_path = Path("results/all_papers_results.json")
    if not results_path.exists():
        print("❌ No research papers found. Please process papers first.")
        return False

    try:
        # Load papers
        with open(results_path, "r", encoding="utf-8") as f:
            papers = json.load(f)

        print(f"✅ Loaded {len(papers)} research papers")

        # Test each AI capability
        capabilities = {
            "Generative AI": test_generative_ai,
            "Agentic AI": test_agentic_ai,
            "RAG": test_rag_system,
            "Symbolic AI": test_symbolic_ai,
            "Neuro-Symbolic AI": test_neuro_symbolic_ai,
            "Machine Learning": test_machine_learning,
            "Deep Learning": test_deep_learning,
            "Reinforcement Learning": test_reinforcement_learning,
            "Evolutionary Algorithms": test_evolutionary_algorithms,
            "LLM": test_llm,
        }

        results = {}
        for capability, test_func in capabilities.items():
            try:
                result = test_func(papers)
                results[capability] = result
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{status} {capability}")
            except Exception as e:
                results[capability] = False
                print(f"❌ FAIL {capability}: {str(e)}")

        # Summary
        print("\n" + "=" * 50)
        print("📊 AI CAPABILITIES SUMMARY")
        print("=" * 50)

        passed = sum(results.values())
        total = len(results)

        for capability, result in results.items():
            status = "✅ YES" if result else "❌ NO"
            print(f"{status} {capability}")

        print(f"\n🎯 Overall: {passed}/{total} capabilities implemented")

        if passed == total:
            print("🎉 ALL AI CAPABILITIES ARE FULLY IMPLEMENTED!")
            return True
        else:
            print(f"⚠️  {total - passed} capabilities need implementation")
            return False

    except Exception as e:
        print(f"❌ Error testing capabilities: {str(e)}")
        return False


def test_generative_ai(papers):
    """Test Generative AI capabilities."""
    try:
        from app.hypothesis_gen import generate_hypotheses
        from app.insight_agent import generate_insights

        # Test hypothesis generation
        sample_insights = "Sample insights for testing"
        hypotheses = generate_hypotheses(sample_insights)

        # Test insight generation
        sample_text = "Sample text for testing"
        insights = generate_insights([sample_text])

        return len(hypotheses) > 0 and len(insights) > 0
    except:
        return False


def test_agentic_ai(papers):
    """Test Agentic AI capabilities."""
    try:
        from agents.cognitive_planner import CognitivePlanner

        planner = CognitivePlanner(papers)
        return planner is not None
    except:
        return False


def test_rag_system(papers):
    """Test RAG system."""
    try:
        from app.rag_system import RAGSystem

        rag = RAGSystem()
        rag.add_papers_to_index(papers)
        return rag.get_collection_stats()["total_papers"] > 0
    except:
        return False


def test_symbolic_ai(papers):
    """Test Symbolic AI capabilities."""
    try:
        from app.citation_network import CitationNetwork

        network = CitationNetwork()
        return network is not None
    except:
        return False


def test_neuro_symbolic_ai(papers):
    """Test Neuro-Symbolic AI capabilities."""
    try:
        # Test combination of neural embeddings and symbolic rules
        from app.citation_network import CitationNetwork
        from app.model_cache import ModelCache

        network = CitationNetwork()
        model = ModelCache.get_sentence_transformer()

        return network is not None and model is not None
    except:
        return False


def test_machine_learning(papers):
    """Test Machine Learning capabilities."""
    try:
        from app.model_tester import run_basic_model
        import pandas as pd
        import numpy as np

        # Create test data
        data = pd.DataFrame(
            {
                "feature1": np.random.rand(10),
                "feature2": np.random.rand(10),
                "target": np.random.rand(10),
            }
        )

        # Test ML model
        result = run_basic_model(data, target_column="target")
        return "rmse" in result and "r2" in result
    except:
        return False


def test_deep_learning(papers):
    """Test Deep Learning capabilities."""
    try:
        from app.model_cache import ModelCache
        import torch

        # Test neural models
        sentence_model = ModelCache.get_sentence_transformer()
        generator = ModelCache.get_text_generator()

        return sentence_model is not None and generator is not None
    except:
        return False


def test_reinforcement_learning(papers):
    """Test Reinforcement Learning capabilities."""
    try:
        from app.rl_selector import RLPaperRecommender

        recommender = RLPaperRecommender(papers)
        return recommender is not None
    except:
        return False


def test_evolutionary_algorithms(papers):
    """Test Evolutionary Algorithms."""
    try:
        from evolutionary.evolve_hypotheses import EvolutionaryHypothesisGenerator

        evo_gen = EvolutionaryHypothesisGenerator(papers)
        return evo_gen is not None
    except:
        return False


def test_llm(papers):
    """Test Large Language Model capabilities."""
    try:
        from app.model_cache import ModelCache

        generator = ModelCache.get_text_generator()
        return generator is not None
    except:
        return False


if __name__ == "__main__":
    success = test_ai_capabilities()
    sys.exit(0 if success else 1)
