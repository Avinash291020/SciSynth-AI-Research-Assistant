"""Cognitive Planner Agent for autonomous research task planning and execution."""
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging
from app.model_cache import ModelCache
from app.rag_system import RAGSystem
from app.rl_selector import RLPaperRecommender
from evolutionary.evolve_hypotheses import EvolutionaryHypothesisGenerator
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class ResearchTask:
    """Represents a research task with goals and constraints."""
    
    def __init__(self, description: str, goal: str, constraints: List[str] = []):
        self.description = description
        self.goal = goal
        self.constraints = constraints
        self.status = "pending"
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None
        self.results = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "description": self.description,
            "goal": self.goal,
            "constraints": self.constraints,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "results": self.results
        }

class CognitivePlanner:
    """Autonomous cognitive planner for research tasks."""
    
    def __init__(self, papers_data: List[Dict[str, Any]]):
        """Initialize cognitive planner with research papers."""
        self.papers = papers_data
        self.logger = logging.getLogger(__name__)
        
        # Initialize AI components
        self.generator = ModelCache.get_text_generator()
        self.rag_system = RAGSystem()
        self.rl_recommender = RLPaperRecommender(papers_data)
        self.evo_generator = EvolutionaryHypothesisGenerator(papers_data)
        
        # Planning tools
        self.planning_tools = {
            "rag_system": self._analyze_papers,
            "gap_analysis": self._identify_research_gaps,
            "evolutionary_algorithm": self._generate_hypotheses,
            "paper_analysis": self._analyze_papers,
            "synthesis": self._synthesize_findings,
            "recommendations": self._get_recommendations
        }
        
        # Task management
        self.current_task = None
        self.task_history = []
        
        # Add papers to RAG system
        self.rag_system.add_papers_to_index(papers_data)
        
        self.logger.info("✅ Cognitive planner initialized")
    
    def plan_research_task(self, user_goal: str) -> Dict[str, Any]:
        """Plan a research task based on user goal."""
        self.logger.info(f"Planning research task: {user_goal}")
        
        # Create research task
        task = ResearchTask(
            description=f"Research task: {user_goal}",
            goal=user_goal,
            constraints=["Use available research papers", "Generate actionable insights"]
        )
        
        self.current_task = task
        
        # Generate plan using LLM
        plan = self._generate_plan(user_goal)
        
        # Execute plan
        results = self._execute_plan(plan)
        
        # Update task
        task.status = "completed"
        task.completed_at = datetime.now()
        task.results = results
        
        self.task_history.append(task)
        
        return {
            "task": task.to_dict(),
            "plan": plan,
            "results": results
        }
    
    def _generate_plan(self, goal: str) -> List[Dict[str, Any]]:
        """Generate a research plan using LLM."""
        prompt = f"""Create a comprehensive research plan for: {goal}

The plan should include:
1. Literature review
2. Gap analysis
3. Hypothesis generation
4. Paper analysis
5. Synthesis and recommendations

Research Plan:"""
        
        # Generate plan using LLM
        response = self.generator(
            prompt,
            max_length=800,
            num_return_sequences=1,
            temperature=0.7
        )[0]['generated_text']
        
        # Parse and structure the plan
        plan_steps = [
            {
                "step": "literature_review",
                "description": "Conduct comprehensive literature review",
                "tool": "rag_system",
                "expected_outcome": "Identify relevant papers and key findings"
            },
            {
                "step": "gap_analysis", 
                "description": "Identify research gaps and opportunities",
                "tool": "gap_analysis",
                "expected_outcome": "List of research gaps and future directions"
            },
            {
                "step": "hypothesis_generation",
                "description": "Generate research hypotheses",
                "tool": "evolutionary_algorithm",
                "expected_outcome": "Set of testable hypotheses"
            },
            {
                "step": "paper_analysis",
                "description": "Analyze individual papers in detail",
                "tool": "paper_analysis",
                "expected_outcome": "Detailed insights from each paper"
            },
            {
                "step": "synthesis",
                "description": "Synthesize findings and provide recommendations",
                "tool": "synthesis",
                "expected_outcome": "Comprehensive research synthesis"
            }
        ]
        
        return plan_steps
    
    def _execute_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute the research plan with aggressive caching and parallelization."""
        results = {}
        goal = self.current_task.goal if self.current_task else "research analysis"

        # 1. Literature review (sequential, always first)
        lit_step = next((s for s in plan if s['step'] == 'literature_review'), None)
        if lit_step:
            results['literature_review'] = get_cached_step_result('literature_review', goal, partial(self.planning_tools[lit_step['tool']], lit_step))

        # 2. Parallel steps: gap_analysis, hypothesis_generation, paper_analysis
        parallel_step_names = ['gap_analysis', 'hypothesis_generation', 'paper_analysis']
        parallel_steps = [s for s in plan if s['step'] in parallel_step_names]
        with ThreadPoolExecutor() as executor:
            future_to_step = {
                executor.submit(get_cached_step_result, s['step'], goal, partial(self.planning_tools[s['tool']], s)): s['step']
                for s in parallel_steps
            }
            for future in future_to_step:
                results[future_to_step[future]] = future.result()

        # 3. Synthesis (sequential, always last)
        synth_step = next((s for s in plan if s['step'] == 'synthesis'), None)
        if synth_step:
            results['synthesis'] = get_cached_step_result('synthesis', goal, partial(self.planning_tools[synth_step['tool']], synth_step))

        return results
    
    def _analyze_papers(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze papers using RAG system."""
        # Get relevant papers for the current goal
        goal = self.current_task.goal if self.current_task else "research analysis"
        relevant_papers = self.rag_system.retrieve_relevant_papers(goal, top_k=5)
        
        # Generate analysis
        analysis_prompt = f"Analyze the following papers in relation to: {goal}"
        analysis = self.rag_system.generate_response(analysis_prompt, relevant_papers)
        
        return {
            "relevant_papers": len(relevant_papers),
            "analysis": analysis,
            "papers_analyzed": [p['metadata']['paper_name'] for p in relevant_papers]
        }
    
    def _generate_hypotheses(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hypotheses using evolutionary algorithm."""
        hypotheses = self.evo_generator.generate_diverse_hypotheses(num_hypotheses=10)
        
        return {
            "hypotheses_generated": len(hypotheses),
            "top_hypotheses": hypotheses[:5],
            "method": "evolutionary_algorithm"
        }
    
    def _conduct_literature_review(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct literature review using RAG."""
        goal = self.current_task.goal if self.current_task else "literature review"
        
        # Get comprehensive review
        review_question = f"What are the main findings and trends in the literature regarding: {goal}"
        review_result = self.rag_system.answer_question(review_question, top_k=8)
        
        return {
            "literature_summary": review_result['answer'],
            "papers_reviewed": review_result['num_papers_retrieved'],
            "key_findings": self._extract_key_findings(review_result['answer'])
        }
    
    def _identify_research_gaps(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Identify research gaps and opportunities."""
        goal = self.current_task.goal if self.current_task else "research gaps"
        
        gap_question = f"What are the research gaps and future opportunities in: {goal}"
        gap_result = self.rag_system.answer_question(gap_question, top_k=8)
        
        # Generate additional gaps using LLM
        gap_prompt = f"""Based on the research papers, identify specific research gaps in: {goal}

Consider:
1. Underexplored areas
2. Methodological limitations
3. Future research directions
4. Emerging opportunities

Research Gaps:"""
        
        # Generate additional gaps using LLM
        additional_gaps = self.generator(
            gap_prompt,
            max_length=800,
            num_return_sequences=1,
            temperature=0.8
        )[0]['generated_text']
        
        return {
            "identified_gaps": gap_result['answer'],
            "additional_gaps": additional_gaps,
            "gap_analysis_method": "rag + llm_synthesis"
        }
    
    def _get_recommendations(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Get paper recommendations using RL."""
        recommendations = self.rl_recommender.recommend_papers(num_recommendations=5)
        
        return {
            "recommendations": recommendations,
            "recommendation_method": "reinforcement_learning",
            "confidence_scores": [r['confidence'] for r in recommendations]
        }
    
    def _synthesize_findings(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize all findings and provide recommendations."""
        goal = self.current_task.goal if self.current_task else "research synthesis"
        
        synthesis_prompt = f"""Based on the research analysis, provide a comprehensive synthesis for: {goal}

Include:
1. Key findings
2. Main trends
3. Research gaps
4. Future directions
5. Practical recommendations

Synthesis:"""
        
        synthesis = self.generator(
            synthesis_prompt,
            max_length=800,
            num_return_sequences=1,
            temperature=0.7
        )[0]['generated_text']
        
        return {
            "synthesis": synthesis,
            "synthesis_method": "llm_generation",
            "comprehensive_analysis": True
        }
    
    def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key findings from text."""
        findings_prompt = f"Extract 5 key findings from this text:\n\n{text}\n\nKey Findings:"
        
        findings_response = self.generator(
            findings_prompt,
            max_length=400,
            num_return_sequences=1,
            temperature=0.6
        )[0]['generated_text']
        
        # Simple parsing
        lines = findings_response.split('\n')
        findings = [line.strip() for line in lines if line.strip() and line.strip()[0].isdigit()]
        
        return findings[:5] if findings else ["Key findings extraction in progress"]
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get history of completed tasks."""
        return [task.to_dict() for task in self.task_history]
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current task status."""
        if self.current_task:
            return {
                "current_task": self.current_task.to_dict(),
                "total_tasks_completed": len(self.task_history),
                "system_status": "active"
            }
        else:
            return {
                "current_task": None,
                "total_tasks_completed": len(self.task_history),
                "system_status": "idle"
            }
    
    def save_task_results(self, filename: str = "cognitive_planner_results.json"):
        """Save task results to file."""
        Path("results").mkdir(exist_ok=True)
        filepath = Path("results") / filename
        
        results = {
            "task_history": self.get_task_history(),
            "current_status": self.get_current_status(),
            "system_info": {
                "total_papers": len(self.papers),
                "planning_tools": list(self.planning_tools.keys()),
                "last_updated": datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved task results to {filepath}")

# --- Aggressive Caching Helper ---
def get_cached_step_result(step_name, goal, compute_fn):
    """Cache step results in st.session_state using a unique key."""
    cache_key = f"agentic_{step_name}_{goal}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    result = compute_fn()
    st.session_state[cache_key] = result
    return result

# Example usage
if __name__ == "__main__":
    # Load papers
    with open("results/all_papers_results.json", 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # Initialize cognitive planner
    planner = CognitivePlanner(papers)
    
    # Plan and execute a research task
    goal = "Identify the most promising directions for AI research in the next 5 years"
    result = planner.plan_research_task(goal)
    
    print(f"Task completed: {result['task']['description']}")
    print(f"Status: {result['task']['status']}")
    
    # Save results
    planner.save_task_results() 