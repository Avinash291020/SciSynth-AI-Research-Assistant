"""Evolutionary Algorithms for hypothesis generation and optimization."""
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from deap import base, creator, tools, algorithms
import json
from pathlib import Path
from app.model_cache import ModelCache

class HypothesisGene:
    """Represents a hypothesis gene in the evolutionary algorithm."""
    
    def __init__(self, components: List[str], relationships: List[str], variables: List[str]):
        self.components = components
        self.relationships = relationships
        self.variables = variables
    
    def to_string(self) -> str:
        """Convert gene to hypothesis string."""
        if not self.components or not self.relationships or not self.variables:
            return "Insufficient components for hypothesis generation."
        
        component = random.choice(self.components)
        relationship = random.choice(self.relationships)
        variable = random.choice(self.variables)
        
        return f"{component} {relationship} {variable}."
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate the hypothesis gene."""
        if random.random() < mutation_rate:
            # Add new component
            new_components = ["neural networks", "deep learning", "reinforcement learning", 
                            "natural language processing", "computer vision", "robotics"]
            self.components.append(random.choice(new_components))
        
        if random.random() < mutation_rate:
            # Add new relationship
            new_relationships = ["improves", "enhances", "reduces", "affects", "influences", 
                               "correlates with", "predicts", "determines"]
            self.relationships.append(random.choice(new_relationships))
        
        if random.random() < mutation_rate:
            # Add new variable
            new_variables = ["performance", "accuracy", "efficiency", "robustness", 
                           "generalization", "interpretability", "scalability"]
            self.variables.append(random.choice(new_variables))

class EvolutionaryHypothesisGenerator:
    """Evolutionary algorithm for generating and optimizing hypotheses."""
    
    def __init__(self, papers_data: List[Dict[str, Any]]):
        self.papers = papers_data
        self.embedding_model = ModelCache.get_sentence_transformer()
        
        # Extract components from papers
        self.components = self._extract_components()
        self.relationships = ["improves", "enhances", "reduces", "affects", "influences", 
                            "correlates with", "predicts", "determines", "enables", "supports"]
        self.variables = ["performance", "accuracy", "efficiency", "robustness", 
                         "generalization", "interpretability", "scalability", "reliability"]
        
        # Initialize DEAP
        self._setup_deap()
    
    def _extract_components(self) -> List[str]:
        """Extract research components from papers."""
        components = set()
        
        for paper in self.papers:
            insights = paper.get('insights', '')
            title = paper.get('title', '')
            
            # Extract key terms
            text = f"{title} {insights}".lower()
            
            # Common AI/ML components
            ai_components = [
                "neural networks", "deep learning", "machine learning", "artificial intelligence",
                "reinforcement learning", "natural language processing", "computer vision",
                "robotics", "optimization", "data mining", "pattern recognition",
                "knowledge representation", "reasoning", "planning", "decision making"
            ]
            
            for component in ai_components:
                if component in text:
                    components.add(component)
        
        return list(components) if components else ["neural networks", "deep learning", "machine learning"]
    
    def _custom_mutator(self, individual: List[Any]) -> Tuple[List[Any],]:
        """Custom mutation function to replace one part of the hypothesis."""
        index_to_mutate = random.randint(0, 2)
        
        if index_to_mutate == 0:
            individual[0] = self.toolbox.component()
        elif index_to_mutate == 1:
            individual[1] = self.toolbox.relationship()
        else:  # index_to_mutate == 2
            individual[2] = self.toolbox.variable()
            
        return individual,

    def _setup_deap(self):
        """Setup DEAP evolutionary algorithm framework."""
        # Create fitness class
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        
        # Create individual class
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Create toolbox
        self.toolbox = base.Toolbox()
        
        # Attribute generators
        self.toolbox.register("component", random.choice, self.components)
        self.toolbox.register("relationship", random.choice, self.relationships)
        self.toolbox.register("variable", random.choice, self.variables)
        
        # Structure initializers
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                            (self.toolbox.component, self.toolbox.relationship, self.toolbox.variable), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("evaluate", self._evaluate_hypothesis)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._custom_mutator)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def _evaluate_hypothesis(self, individual) -> Tuple[float]:
        """Evaluate the fitness of a hypothesis."""
        if not individual:
            return (0.0,)
        
        # Create hypothesis string
        hypothesis = f"{individual[0]} {individual[1]} {individual[2]}."
        
        # Calculate fitness based on relevance to papers
        fitness = self._calculate_relevance(hypothesis)
        
        return (fitness,)
    
    def _calculate_relevance(self, hypothesis: str) -> float:
        """Calculate relevance of hypothesis to research papers."""
        if not self.papers:
            return 0.5  # Default score
        
        # Encode hypothesis
        hypothesis_embedding = self.embedding_model.encode([hypothesis])
        
        # Calculate similarity with all papers
        similarities = []
        for paper in self.papers:
            paper_text = f"{paper.get('title', '')} {paper.get('insights', '')}"
            if paper_text.strip():
                paper_embedding = self.embedding_model.encode([paper_text])
                similarity = np.dot(hypothesis_embedding[0], paper_embedding[0]) / (
                    np.linalg.norm(hypothesis_embedding[0]) * np.linalg.norm(paper_embedding[0])
                )
                similarities.append(similarity)
        
        # Return average similarity
        return np.mean(similarities) if similarities else 0.0
    
    def evolve_hypotheses(self, population_size: int = 50, generations: int = 20) -> List[Dict[str, Any]]:
        """Evolve hypotheses using genetic algorithm."""
        # Create initial population
        pop = self.toolbox.population(n=population_size)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Hall of fame
        hof = tools.HallOfFame(10)
        
        # Run evolution
        pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=0.7, mutpb=0.2, 
                                         ngen=generations, stats=stats, halloffame=hof, verbose=True)
        
        # Convert results to readable format
        evolved_hypotheses = []
        for i, individual in enumerate(hof):
            hypothesis = f"{individual[0]} {individual[1]} {individual[2]}."
            fitness = individual.fitness.values[0]
            
            evolved_hypotheses.append({
                'hypothesis': hypothesis,
                'fitness': fitness,
                'rank': i + 1,
                'components': list(individual)
            })
        
        return evolved_hypotheses
    
    def generate_diverse_hypotheses(self, num_hypotheses: int = 10) -> List[Dict[str, Any]]:
        """Generate diverse hypotheses using evolutionary approach."""
        # Run evolution
        evolved = self.evolve_hypotheses(population_size=30, generations=15)
        
        # Add some random hypotheses for diversity
        diverse_hypotheses = evolved[:num_hypotheses//2]
        
        for _ in range(num_hypotheses - len(diverse_hypotheses)):
            component = random.choice(self.components)
            relationship = random.choice(self.relationships)
            variable = random.choice(self.variables)
            
            hypothesis = f"{component} {relationship} {variable}."
            fitness = self._calculate_relevance(hypothesis)
            
            diverse_hypotheses.append({
                'hypothesis': hypothesis,
                'fitness': fitness,
                'rank': len(diverse_hypotheses) + 1,
                'components': [component, relationship, variable]
            })
        
        # Sort by fitness
        diverse_hypotheses.sort(key=lambda x: x['fitness'], reverse=True)
        
        return diverse_hypotheses
    
    def optimize_hypothesis(self, initial_hypothesis: str, iterations: int = 100) -> Dict[str, Any]:
        """Optimize a specific hypothesis using evolutionary approach."""
        # Parse initial hypothesis
        words = initial_hypothesis.split()
        if len(words) < 3:
            return {'hypothesis': initial_hypothesis, 'fitness': 0.0, 'optimized': False}
        
        # Create initial individual
        initial_individual = creator.Individual([words[0], words[1], words[2]])
        
        # Setup optimization
        self.toolbox.register("evaluate", self._evaluate_hypothesis)
        
        # Run optimization
        best_individual = initial_individual
        best_fitness = self.toolbox.evaluate(initial_individual)[0]
        
        for _ in range(iterations):
            # Create offspring
            offspring = self.toolbox.clone(best_individual)
            
            # Mutate
            self.toolbox.mutate(offspring)
            
            # Evaluate
            fitness = self.toolbox.evaluate(offspring)[0]
            
            # Update if better
            if fitness > best_fitness:
                best_individual = offspring
                best_fitness = fitness
        
        optimized_hypothesis = f"{best_individual[0]} {best_individual[1]} {best_individual[2]}."
        
        return {
            'original_hypothesis': initial_hypothesis,
            'optimized_hypothesis': optimized_hypothesis,
            'original_fitness': self._evaluate_hypothesis(initial_individual)[0],
            'optimized_fitness': best_fitness,
            'improvement': best_fitness - self._evaluate_hypothesis(initial_individual)[0],
            'optimized': True
        }
    
    def save_results(self, hypotheses: List[Dict[str, Any]], filename: str = "evolved_hypotheses.json"):
        """Save evolved hypotheses to file."""
        Path("results").mkdir(exist_ok=True)
        filepath = Path("results") / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(hypotheses, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(hypotheses)} evolved hypotheses to {filepath}")

# Example usage
if __name__ == "__main__":
    # Load papers
    with open("results/all_papers_results.json", 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # Initialize evolutionary hypothesis generator
    evo_gen = EvolutionaryHypothesisGenerator(papers)
    
    # Generate diverse hypotheses
    print("Generating diverse hypotheses...")
    hypotheses = evo_gen.generate_diverse_hypotheses(num_hypotheses=15)
    
    print("\nTop evolved hypotheses:")
    for i, hyp in enumerate(hypotheses[:5], 1):
        print(f"{i}. {hyp['hypothesis']} (fitness: {hyp['fitness']:.3f})")
    
    # Optimize a specific hypothesis
    print("\nOptimizing hypothesis...")
    initial_hyp = "neural networks improve performance"
    optimization_result = evo_gen.optimize_hypothesis(initial_hyp)
    
    print(f"Original: {optimization_result['original_hypothesis']}")
    print(f"Optimized: {optimization_result['optimized_hypothesis']}")
    print(f"Improvement: {optimization_result['improvement']:.3f}")
    
    # Save results
    evo_gen.save_results(hypotheses) 