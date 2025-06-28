# -*- coding: utf-8 -*-
"""
Test suite for all AI paradigms.
"""

import json
import traceback
from pathlib import Path

from app.rag_system import RAGSystem
from app.rl_selector import RLPaperRecommender
from evolutionary.evolve_hypotheses import EvolutionaryHypothesisGenerator
from agents.cognitive_planner import CognitivePlanner
from agents.orchestrator import ResearchOrchestrator

# Optional: Symbolic/Neuro-Symbolic
try:
    from logic.consistency_checker import check_consistency

    symbolic_available = True
except ImportError:
    symbolic_available = False


def print_header(title):
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)


def main():
    # Load research papers
    papers_path = Path("results/all_papers_results.json")
    if not papers_path.exists():
        print("ERROR: results/all_papers_results.json not found.")
        return
    with open(papers_path, "r", encoding="utf-8") as f:
        papers = json.load(f)
    print(f"Loaded {len(papers)} papers.")

    # 1. RAG System
    print_header("RAG System Test")
    try:
        rag = RAGSystem()
        rag.add_papers_to_index(papers)
        question = "What are the key advances in neuro-symbolic AI?"
        result = rag.answer_question(question, top_k=3)
        print("RAG Answer:", result["answer"][:300], "...\n")
        print(
            "Relevant papers:",
            [p["metadata"].get("paper_name") for p in result["relevant_papers"]],
        )
        print("PASS: RAG system works.")
    except Exception as e:
        print("FAIL: RAG system error:", e)
        traceback.print_exc()

    # 2. RL Paper Recommender
    print_header("RL Paper Recommender Test")
    try:
        rl = RLPaperRecommender(papers)
        recs = rl.recommend_papers(num_recommendations=3)
        print("RL Recommendations:", recs)
        print("PASS: RL recommender works.")
    except Exception as e:
        print("FAIL: RL recommender error:", e)
        traceback.print_exc()

    # 3. Evolutionary Hypothesis Generator
    print_header("Evolutionary Hypothesis Generator Test")
    try:
        evo = EvolutionaryHypothesisGenerator(papers)
        hyps = evo.generate_diverse_hypotheses(num_hypotheses=3)
        print("Evolutionary Hypotheses:", hyps)
        print("PASS: Evolutionary generator works.")
    except Exception as e:
        print("FAIL: Evolutionary generator error:", e)
        traceback.print_exc()

    # 4. Cognitive Planner
    print_header("Cognitive Planner Test")
    try:
        planner = CognitivePlanner(papers)
        plan_result = planner.plan_research_task(
            "How can reinforcement learning improve scientific discovery?"
        )
        print("Cognitive Plan:", plan_result["plan"])
        print("Cognitive Results:", plan_result["results"])
        print("PASS: Cognitive planner works.")
    except Exception as e:
        print("FAIL: Cognitive planner error:", e)
        traceback.print_exc()

    # 5. Orchestrator (All Systems)
    print_header("Orchestrator (All Systems) Test")
    try:
        orchestrator = ResearchOrchestrator(papers)
        analysis = orchestrator.comprehensive_research_analysis(
            "What are the main trends in AI research?"
        )
        print("Orchestrator Synthesis:", analysis.get("final_synthesis", {}))
        print("PASS: Orchestrator works.")
    except Exception as e:
        print("FAIL: Orchestrator error:", e)
        traceback.print_exc()

    # 6. Symbolic/Neuro-Symbolic (if available)
    if symbolic_available:
        print_header("Symbolic/Neuro-Symbolic AI Test")
        try:
            # Example: check consistency of a simple rule
            result = check_consistency("A implies B and B implies C")
            print("Symbolic Consistency Result:", result)
            print("PASS: Symbolic/Neuro-Symbolic AI works.")
        except Exception as e:
            print("FAIL: Symbolic/Neuro-Symbolic AI error:", e)
            traceback.print_exc()
    else:
        print(
            "Symbolic/Neuro-Symbolic AI not available (consistency_checker not found)."
        )


if __name__ == "__main__":
    main()
