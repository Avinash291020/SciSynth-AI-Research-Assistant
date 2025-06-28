# -*- coding: utf-8 -*-
"""
Generate citation network visualizations.
"""

#!/usr/bin/env python3
"""Generate citation network analysis for existing papers."""

import json
from pathlib import Path
from app.citation_network import CitationNetwork
import networkx as nx


def main():
    print("Generating citation network analysis...")

    # Load existing results
    results_path = Path("results/all_papers_results.json")
    if not results_path.exists():
        print("Error: all_papers_results.json not found!")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    print(f"Loaded {len(all_results)} papers")

    # Create citation network manually
    network = CitationNetwork()

    # Build citation network
    for paper in all_results:
        network.graph.add_node(
            paper["paper_name"],
            title=(
                paper.get("insights", "").split("\n")[1]
                if paper.get("insights")
                else "Unknown"
            ),
            date=paper["processed_date"],
        )

        # Extract citations from insights and hypotheses
        all_text = paper.get("insights", "") + "\n" + paper.get("hypotheses", "")
        citations = network.extract_citations(all_text)

        for citation in citations:
            network.graph.add_edge(paper["paper_name"], f"Citation: {citation}")

    # Compute paper relationships
    relationships = []
    for i, paper1 in enumerate(all_results):
        for paper2 in all_results[i + 1 :]:
            similarity = network.compute_similarity(
                paper1.get("insights", ""), paper2.get("insights", "")
            )
            if similarity > 0.3:  # Lower threshold to get more relationships
                relationships.append(
                    {
                        "paper1": paper1["paper_name"],
                        "paper2": paper2["paper_name"],
                        "similarity": similarity,
                        "shared_topics": network._find_shared_topics(
                            paper1.get("insights", ""), paper2.get("insights", "")
                        ),
                    }
                )

    # Create network analysis results
    network_results = {
        "num_papers": len(all_results),
        "num_citations": len(
            [n for n in network.graph.nodes() if n.startswith("Citation:")]
        ),
        "num_relationships": len(relationships),
        "relationships": relationships,
        "cross_references": {"direct_citations": [], "shared_topics": []},
    }

    # Save network analysis
    with open(Path("results/network_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(network_results, f, indent=2, ensure_ascii=False)

    # Generate visualization
    network.visualize_network(str(Path("results/citation_network.png")))

    print("Citation network analysis completed!")
    print(f"- Network analysis saved to: results/network_analysis.json")
    print(f"- Visualization saved to: results/citation_network.png")
    print(
        f"- Found {network_results['num_relationships']} relationships between papers"
    )
    print(f"- Found {network_results['num_citations']} citations")


if __name__ == "__main__":
    main()
