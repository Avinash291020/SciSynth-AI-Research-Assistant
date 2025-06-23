"""Orchestrator for coordinating all AI systems in the research assistant."""
import json
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging
from app.rag_system import RAGSystem
from app.rl_selector import RLPaperRecommender
from evolutionary.evolve_hypotheses import EvolutionaryHypothesisGenerator
from agents.cognitive_planner import CognitivePlanner
from app.model_cache import ModelCache

class ResearchOrchestrator:
    """Main orchestrator that coordinates all AI systems."""
    
    def __init__(self, papers_data: List[Dict[str, Any]]):
        self.papers = papers_data
        self.generator = ModelCache.get_text_generator()
        
        # Initialize all AI systems
        self.rag_system = RAGSystem()
        self.rl_recommender = RLPaperRecommender(papers_data)
        self.evo_generator = EvolutionaryHypothesisGenerator(papers_data)
        self.cognitive_planner = CognitivePlanner(papers_data)
        
        # System status
        self.systems_status = {
            "rag": "initialized",
            "rl": "initialized", 
            "evolutionary": "initialized",
            "cognitive_planner": "initialized"
        }
        
        # Setup logger first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize systems
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize all AI systems."""
        try:
            # Initialize RAG with papers
            self.rag_system.add_papers_to_index(self.papers)
            self.systems_status["rag"] = "ready"
            self.logger.info("✅ RAG system initialized")
            
            # RL system is ready (no training needed initially)
            self.systems_status["rl"] = "ready"
            self.logger.info("✅ RL system initialized")
            
            # Evolutionary system is ready
            self.systems_status["evolutionary"] = "ready"
            self.logger.info("✅ Evolutionary system initialized")
            
            # Cognitive planner is ready
            self.systems_status["cognitive_planner"] = "ready"
            self.logger.info("✅ Cognitive planner initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing systems: {str(e)}")
    
    def comprehensive_research_analysis(self, research_question: str) -> Dict[str, Any]:
        """Perform comprehensive research analysis using all AI systems."""
        self.logger.info(f"Starting comprehensive analysis for: {research_question}")
        
        results = {
            "research_question": research_question,
            "timestamp": datetime.now().isoformat(),
            "systems_used": list(self.systems_status.keys()),
            "analysis_results": {}
        }
        
        # 1. RAG-based literature review
        try:
            rag_results = self.rag_system.answer_question(research_question, top_k=5)
            results["analysis_results"]["rag_literature_review"] = {
                "answer": rag_results["answer"],
                "papers_retrieved": rag_results["num_papers_retrieved"],
                "relevant_papers": [p["metadata"]["paper_name"] for p in rag_results["relevant_papers"]]
            }
            self.logger.info("✅ RAG literature review completed")
        except Exception as e:
            self.logger.error(f"RAG error: {str(e)}")
            results["analysis_results"]["rag_literature_review"] = {"error": str(e)}
        
        # 2. RL-based paper recommendations
        try:
            recommendations = self.rl_recommender.recommend_papers(num_recommendations=5)
            results["analysis_results"]["rl_recommendations"] = {
                "recommendations": recommendations,
                "recommendation_method": "reinforcement_learning"
            }
            self.logger.info("✅ RL recommendations completed")
        except Exception as e:
            self.logger.error(f"RL error: {str(e)}")
            results["analysis_results"]["rl_recommendations"] = {"error": str(e)}
        
        # 3. Evolutionary hypothesis generation
        try:
            hypotheses = self.evo_generator.generate_diverse_hypotheses(num_hypotheses=8)
            results["analysis_results"]["evolutionary_hypotheses"] = {
                "hypotheses": hypotheses[:5],  # Top 5
                "total_generated": len(hypotheses),
                "method": "evolutionary_algorithm"
            }
            self.logger.info("✅ Evolutionary hypothesis generation completed")
        except Exception as e:
            self.logger.error(f"Evolutionary error: {str(e)}")
            results["analysis_results"]["evolutionary_hypotheses"] = {"error": str(e)}
        
        # 4. Cognitive planning and synthesis
        try:
            cognitive_results = self.cognitive_planner.plan_research_task(research_question)
            results["analysis_results"]["cognitive_planning"] = {
                "task_results": cognitive_results["results"],
                "plan_executed": cognitive_results["plan"],
                "method": "autonomous_cognitive_planning"
            }
            self.logger.info("✅ Cognitive planning completed")
        except Exception as e:
            self.logger.error(f"Cognitive planning error: {str(e)}")
            results["analysis_results"]["cognitive_planning"] = {"error": str(e)}
        
        # 5. Generate final synthesis
        try:
            final_synthesis = self._generate_final_synthesis(results["analysis_results"])
            results["final_synthesis"] = final_synthesis
            self.logger.info("✅ Final synthesis completed")
        except Exception as e:
            self.logger.error(f"Synthesis error: {str(e)}")
            results["final_synthesis"] = {"error": str(e)}
        
        return results
    
    def _generate_final_synthesis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        synthesis_prompt = """Synthesize the following research analysis results:

1. RAG Literature Review: {rag_review}
2. RL Recommendations: {rl_recs}
3. Evolutionary Hypotheses: {evo_hypotheses}
4. Cognitive Planning: {cognitive_results}

Generate a comprehensive synthesis that includes:
- Key findings and insights
- Research gaps and opportunities
- Future research directions
- Practical recommendations
- Confidence levels in different findings

Final Synthesis:"""
        rag_review = analysis_results.get("rag_literature_review", {}).get("answer", "No RAG results available")
        rl_recs = str(analysis_results.get("rl_recommendations", {}).get("recommendations", []))
        evo_hypotheses = str(analysis_results.get("evolutionary_hypotheses", {}).get("hypotheses", []))
        cognitive_results = str(analysis_results.get("cognitive_planning", {}).get("task_results", {}))
        full_prompt = synthesis_prompt.format(
            rag_review=rag_review[:300],
            rl_recs=rl_recs[:200],
            evo_hypotheses=evo_hypotheses[:200],
            cognitive_results=cognitive_results[:200]
        )
        synthesis = self.generator(
            full_prompt,
            max_length=800,
            num_return_sequences=1,
            temperature=0.7
        )[0]['generated_text']
        return {
            "synthesis": synthesis,
            "synthesis_method": "multi_ai_system_integration",
            "confidence": "high",
            "systems_integrated": len(analysis_results)
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all AI systems."""
        return {
            "systems_status": self.systems_status,
            "total_papers": len(self.papers),
            "orchestrator_status": "active",
            "last_updated": datetime.now().isoformat()
        }
    
    def train_rl_system(self, episodes: int = 50) -> Dict[str, Any]:
        """Train the RL recommendation system."""
        try:
            self.logger.info("Training RL system...")
            scores = self.rl_recommender.train(episodes=episodes)
            
            # Save trained model
            self.rl_recommender.save_model()
            
            return {
                "training_completed": True,
                "episodes": episodes,
                "final_score": scores[-1] if scores else 0,
                "average_score": sum(scores) / len(scores) if scores else 0,
                "model_saved": True
            }
        except Exception as e:
            self.logger.error(f"RL training error: {str(e)}")
            return {"training_completed": False, "error": str(e)}
    
    def ask_question(self, question: str, use_all_systems: bool = True) -> Dict[str, Any]:
        """Ask a question using the orchestrator."""
        if use_all_systems:
            return self.comprehensive_research_analysis(question)
        else:
            # Use only RAG for simple questions
            return {
                "question": question,
                "answer": self.rag_system.answer_question(question),
                "method": "rag_only"
            }
    
    def generate_research_report(self, topic: str) -> Dict[str, Any]:
        """Generate a comprehensive research report."""
        self.logger.info(f"Generating research report for: {topic}")
        
        # Perform comprehensive analysis
        analysis = self.comprehensive_research_analysis(topic)
        
        # Generate report structure
        report = {
            "title": f"Research Report: {topic}",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": analysis["final_synthesis"]["synthesis"][:500] + "...",
            "sections": {
                "literature_review": analysis["analysis_results"].get("rag_literature_review", {}),
                "recommendations": analysis["analysis_results"].get("rl_recommendations", {}),
                "hypotheses": analysis["analysis_results"].get("evolutionary_hypotheses", {}),
                "research_plan": analysis["analysis_results"].get("cognitive_planning", {}),
                "synthesis": analysis["final_synthesis"]
            },
            "methodology": {
                "rag_system": "Retrieval-Augmented Generation for literature review",
                "rl_system": "Reinforcement Learning for paper recommendations",
                "evolutionary_system": "Evolutionary Algorithms for hypothesis generation",
                "cognitive_system": "Autonomous cognitive planning and synthesis"
            }
        }
        
        return report
    
    def save_results(self, results: Dict[str, Any], filename: str = "orchestrator_results.json"):
        """Save orchestrator results to file."""
        Path("results").mkdir(exist_ok=True)
        filepath = Path("results") / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ Saved orchestrator results to {filepath}")
    
    def get_ai_capabilities_summary(self) -> Dict[str, Any]:
        """Get summary of all AI capabilities."""
        return {
            "generative_ai": {
                "status": "active",
                "capabilities": ["text generation", "hypothesis generation", "synthesis"],
                "models": ["T5", "Sentence Transformers"]
            },
            "agentic_ai": {
                "status": "active", 
                "capabilities": ["autonomous planning", "task execution", "goal-oriented behavior"],
                "components": ["cognitive planner", "orchestrator"]
            },
            "rag": {
                "status": self.systems_status["rag"],
                "capabilities": ["document retrieval", "contextual generation", "question answering"],
                "database": "ChromaDB"
            },
            "symbolic_ai": {
                "status": "active",
                "capabilities": ["rule-based reasoning", "logical inference", "structured analysis"],
                "components": ["citation extraction", "network analysis"]
            },
            "neuro_symbolic_ai": {
                "status": "active",
                "capabilities": ["neural-symbolic integration", "hybrid reasoning"],
                "components": ["RAG + symbolic rules", "neural embeddings + logical analysis"]
            },
            "machine_learning": {
                "status": "active",
                "capabilities": ["supervised learning", "unsupervised learning", "embedding generation"],
                "models": ["Sentence Transformers", "scikit-learn"]
            },
            "deep_learning": {
                "status": "active",
                "capabilities": ["neural networks", "transformers", "deep representations"],
                "models": ["T5", "Sentence Transformers", "PyTorch"]
            },
            "reinforcement_learning": {
                "status": self.systems_status["rl"],
                "capabilities": ["Q-learning", "policy optimization", "recommendation systems"],
                "components": ["DQN agent", "paper recommendation"]
            },
            "evolutionary_algorithms": {
                "status": self.systems_status["evolutionary"],
                "capabilities": ["genetic algorithms", "hypothesis evolution", "optimization"],
                "components": ["DEAP framework", "hypothesis generation"]
            },
            "llm": {
                "status": "active",
                "capabilities": ["text generation", "comprehension", "reasoning"],
                "models": ["T5", "local transformers"]
            }
        }

# Example usage
if __name__ == "__main__":
    # Load papers
    with open("results/all_papers_results.json", 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # Initialize orchestrator
    orchestrator = ResearchOrchestrator(papers)
    
    # Get system status
    status = orchestrator.get_system_status()
    print("System Status:", status)
    
    # Perform comprehensive analysis
    question = "What are the emerging trends in artificial intelligence research?"
    results = orchestrator.comprehensive_research_analysis(question)
    
    # Generate research report
    report = orchestrator.generate_research_report("AI Research Trends")
    
    # Save results
    orchestrator.save_results(results)
    
    # Get AI capabilities summary
    capabilities = orchestrator.get_ai_capabilities_summary()
    print("\nAI Capabilities Summary:")
    for ai_type, details in capabilities.items():
        print(f"{ai_type.upper()}: {details['status']} - {', '.join(details['capabilities'])}") 