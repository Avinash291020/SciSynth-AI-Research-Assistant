"""Citation network analysis for research papers."""
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
import re

class CitationNetwork:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract citations from text using regex."""
        # Look for citations in various formats
        patterns = [
            r'\[([^\]]+)\]',  # [Author et al. 2023]
            r'\(([^)]+\d{4}[^)]*)\)',  # (Author et al. 2023)
            r'(?:see|in|by|from)\s+([A-Z][a-z]+\s+et\s+al\.\s+\d{4})',  # see Author et al. 2023
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            citations.extend(match.group(1) for match in matches)
        
        return list(set(citations))  # Remove duplicates
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        emb1 = self.model.encode(text1, convert_to_tensor=True)
        emb2 = self.model.encode(text2, convert_to_tensor=True)
        return float(emb1 @ emb2.T)
    
    def _find_shared_topics(self, text1: str, text2: str) -> List[str]:
        """Find shared topics between two texts."""
        # Simple keyword extraction and matching
        keywords1 = set(re.findall(r'\b[A-Z][a-z]{2,}\w*(?:\s+[A-Z][a-z]{2,}\w*)*', text1))
        keywords2 = set(re.findall(r'\b[A-Z][a-z]{2,}\w*(?:\s+[A-Z][a-z]{2,}\w*)*', text2))
        return list(keywords1.intersection(keywords2))
    
    def analyze_papers(self, results_dir: Path) -> Tuple[nx.DiGraph, List[Dict]]:
        """Analyze citation network and relationships between papers."""
        # Load all paper results
        papers = []
        for result_file in results_dir.glob("*_results.json"):
            with open(result_file, 'r', encoding='utf-8') as f:
                papers.append(json.load(f))
        
        # Build citation network
        for paper in papers:
            self.graph.add_node(paper["paper_name"], 
                              title=paper.get("insights", "").split("\n")[1],  # First line after "Title:"
                              date=paper["processed_date"])
            
            # Extract citations from insights and hypotheses
            all_text = paper["insights"] + "\n" + paper["hypotheses"]
            citations = self.extract_citations(all_text)
            
            for citation in citations:
                self.graph.add_edge(paper["paper_name"], f"Citation: {citation}")
        
        # Compute paper relationships
        relationships = []
        for i, paper1 in enumerate(papers):
            for paper2 in papers[i+1:]:
                similarity = self.compute_similarity(paper1["insights"], paper2["insights"])
                if similarity > 0.5:  # Only include significant relationships
                    relationships.append({
                        "paper1": paper1["paper_name"],
                        "paper2": paper2["paper_name"],
                        "similarity": similarity,
                        "shared_topics": self._find_shared_topics(paper1["insights"], paper2["insights"])
                    })
        
        return self.graph, relationships
    
    def visualize_network(self, output_path: str):
        """Generate and save a visualization of the citation network."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        # Draw papers
        paper_nodes = [n for n in self.graph.nodes() if not n.startswith("Citation:")]
        nx.draw_networkx_nodes(self.graph, pos, nodelist=paper_nodes, 
                             node_color='lightblue', node_size=1000)
        
        # Draw citations
        citation_nodes = [n for n in self.graph.nodes() if n.startswith("Citation:")]
        nx.draw_networkx_nodes(self.graph, pos, nodelist=citation_nodes,
                             node_color='lightgreen', node_size=500)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True)
        
        # Add labels
        labels = {n: n.replace("Citation: ", "") for n in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title("Research Paper Citation Network")
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close() 