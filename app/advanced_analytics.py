"""Advanced Analytics Module for Research Paper Analysis."""
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

class AdvancedAnalytics:
    def __init__(self, papers_data: List[Dict[str, Any]]):
        """Initialize analytics with enriched paper data."""
        self.papers = papers_data
        self.metadata_list = [p.get('metadata', {}) for p in papers_data]
    
    def get_author_network(self) -> Dict[str, Any]:
        """Generate author collaboration network."""
        # Create author-paper mapping
        author_papers = defaultdict(list)
        for i, metadata in enumerate(self.metadata_list):
            authors = metadata.get('authors', [])
            for author in authors:
                author_papers[author].append(i)
        
        # Create collaboration edges
        collaborations = defaultdict(int)
        for author, papers in author_papers.items():
            for i, paper1 in enumerate(papers):
                for paper2 in papers[i+1:]:
                    # Find co-authors of paper1
                    co_authors1 = set(self.metadata_list[paper1].get('authors', []))
                    co_authors2 = set(self.metadata_list[paper2].get('authors', []))
                    
                    # Add collaborations between co-authors
                    for author1 in co_authors1:
                        for author2 in co_authors2:
                            if author1 != author2:
                                edge = tuple(sorted([author1, author2]))
                                collaborations[edge] += 1
        
        # Create network graph
        G = nx.Graph()
        for (author1, author2), weight in collaborations.items():
            G.add_edge(author1, author2, weight=weight)
        
        # Calculate network metrics
        metrics = {
            'total_authors': len(author_papers),
            'total_collaborations': len(collaborations),
            'connected_components': nx.number_connected_components(G),
            'average_clustering': nx.average_clustering(G) if len(G) > 0 else 0,
            'density': nx.density(G) if len(G) > 0 else 0
        }
        
        # Get top collaborators
        top_collaborations = sorted(collaborations.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'graph': G,
            'metrics': metrics,
            'top_collaborations': top_collaborations,
            'author_papers': dict(author_papers)
        }
    
    def get_citation_analysis(self) -> Dict[str, Any]:
        """Analyze citations and references."""
        citation_stats = {
            'papers_with_refs': 0,
            'total_references': 0,
            'papers_with_dois': 0,
            'papers_with_urls': 0,
            'avg_refs_per_paper': 0,
            'reference_years': [],
            'common_venues': []
        }
        
        all_references = []
        venue_counts = Counter()
        year_counts = Counter()
        
        for metadata in self.metadata_list:
            refs = metadata.get('references', [])
            if refs:
                citation_stats['papers_with_refs'] += 1
                citation_stats['total_references'] += len(refs)
                all_references.extend(refs)
            
            dois = metadata.get('dois', [])
            if dois:
                citation_stats['papers_with_dois'] += 1
            
            urls = metadata.get('urls', [])
            if urls:
                citation_stats['papers_with_urls'] += 1
        
        # Calculate averages
        if citation_stats['papers_with_refs'] > 0:
            citation_stats['avg_refs_per_paper'] = citation_stats['total_references'] / citation_stats['papers_with_refs']
        
        # Analyze reference years
        for ref in all_references:
            if isinstance(ref, dict) and ref.get('year'):
                year_counts[ref['year']] += 1
        
        citation_stats['reference_years'] = dict(year_counts.most_common(10))
        
        return citation_stats
    
    def get_section_analysis(self) -> Dict[str, Any]:
        """Analyze paper sections and content."""
        section_stats = {
            'papers_with_sections': 0,
            'section_frequency': Counter(),
            'avg_sections_per_paper': 0,
            'section_content_length': defaultdict(list)
        }
        
        total_sections = 0
        
        for metadata in self.metadata_list:
            sections = metadata.get('sections', {})
            if sections:
                section_stats['papers_with_sections'] += 1
                total_sections += len(sections)
                
                for section_name, section_text in sections.items():
                    section_stats['section_frequency'][section_name] += 1
                    section_stats['section_content_length'][section_name].append(len(section_text))
        
        if section_stats['papers_with_sections'] > 0:
            section_stats['avg_sections_per_paper'] = total_sections / section_stats['papers_with_sections']
        
        # Calculate average content length per section
        for section_name, lengths in section_stats['section_content_length'].items():
            if lengths:
                section_stats['section_content_length'][section_name] = sum(lengths) / len(lengths)
        
        return section_stats
    
    def get_topic_evolution(self) -> Dict[str, Any]:
        """Analyze how topics evolve over time."""
        topic_evolution = defaultdict(lambda: defaultdict(int))
        
        for metadata in self.metadata_list:
            year = metadata.get('date', '')
            keywords = metadata.get('keywords', [])
            technical_terms = metadata.get('technical_terms', [])
            
            if year and year.isdigit():
                for keyword in keywords:
                    topic_evolution[keyword][year] += 1
                for term in technical_terms:
                    topic_evolution[term][year] += 1
        
        # Get top topics
        topic_totals = {}
        for topic, year_counts in topic_evolution.items():
            topic_totals[topic] = sum(year_counts.values())
        
        top_topics = sorted(topic_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'topic_evolution': dict(topic_evolution),
            'top_topics': top_topics
        }
    
    def get_institution_analysis(self) -> Dict[str, Any]:
        """Analyze institutional collaborations and contributions."""
        institution_stats = {
            'total_institutions': 0,
            'institution_papers': defaultdict(list),
            'top_institutions': [],
            'institution_collaborations': defaultdict(set)
        }
        
        institution_papers = defaultdict(list)
        
        for i, metadata in enumerate(self.metadata_list):
            affiliations = metadata.get('affiliations', [])
            if affiliations:
                institution_stats['total_institutions'] += len(affiliations)
                
                for affil in affiliations:
                    institution_papers[affil].append(i)
                    institution_stats['institution_papers'][affil].append(i)
        
        # Find top institutions
        institution_counts = {affil: len(papers) for affil, papers in institution_papers.items()}
        institution_stats['top_institutions'] = sorted(institution_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Find institutional collaborations
        for i, metadata in enumerate(self.metadata_list):
            affiliations = metadata.get('affiliations', [])
            for j, affil1 in enumerate(affiliations):
                for affil2 in affiliations[j+1:]:
                    if affil1 != affil2:
                        institution_stats['institution_collaborations'][affil1].add(affil2)
                        institution_stats['institution_collaborations'][affil2].add(affil1)
        
        return institution_stats
    
    def get_paper_type_analysis(self) -> Dict[str, Any]:
        """Analyze distribution and characteristics of different paper types."""
        type_analysis = {
            'type_distribution': Counter(),
            'type_keywords': defaultdict(list),
            'type_authors': defaultdict(list),
            'type_sections': defaultdict(list)
        }
        
        for metadata in self.metadata_list:
            paper_type = metadata.get('paper_type', 'unknown')
            type_analysis['type_distribution'][paper_type] += 1
            
            # Collect keywords by type
            keywords = metadata.get('keywords', [])
            if keywords:
                type_analysis['type_keywords'][paper_type].extend(keywords)
            
            # Collect authors by type
            authors = metadata.get('authors', [])
            if authors:
                type_analysis['type_authors'][paper_type].extend(authors)
            
            # Collect sections by type
            sections = metadata.get('sections', {})
            if sections:
                type_analysis['type_sections'][paper_type].extend(list(sections.keys()))
        
        # Get most common keywords per type
        for paper_type, keywords in type_analysis['type_keywords'].items():
            keyword_counts = Counter(keywords)
            type_analysis['type_keywords'][paper_type] = keyword_counts.most_common(5)
        
        # Get most common authors per type
        for paper_type, authors in type_analysis['type_authors'].items():
            author_counts = Counter(authors)
            type_analysis['type_authors'][paper_type] = author_counts.most_common(5)
        
        # Get most common sections per type
        for paper_type, sections in type_analysis['type_sections'].items():
            section_counts = Counter(sections)
            type_analysis['type_sections'][paper_type] = section_counts.most_common(5)
        
        return type_analysis
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analytics report."""
        report = {
            'author_network': self.get_author_network(),
            'citation_analysis': self.get_citation_analysis(),
            'section_analysis': self.get_section_analysis(),
            'topic_evolution': self.get_topic_evolution(),
            'institution_analysis': self.get_institution_analysis(),
            'paper_type_analysis': self.get_paper_type_analysis(),
            'summary': {
                'total_papers': len(self.papers),
                'papers_with_metadata': sum(1 for m in self.metadata_list if m.get('title')),
                'papers_with_authors': sum(1 for m in self.metadata_list if m.get('authors')),
                'papers_with_affiliations': sum(1 for m in self.metadata_list if m.get('affiliations')),
                'papers_with_sections': sum(1 for m in self.metadata_list if m.get('sections')),
                'papers_with_references': sum(1 for m in self.metadata_list if m.get('references')),
                'papers_with_dois': sum(1 for m in self.metadata_list if m.get('dois')),
                'papers_with_urls': sum(1 for m in self.metadata_list if m.get('urls'))
            }
        }
        
        return report
    
    def save_analytics_report(self, output_path: str = "results/advanced_analytics_report.json"):
        """Save the comprehensive analytics report to a JSON file."""
        report = self.generate_comprehensive_report()
        
        # Convert networkx graph to serializable format
        if 'graph' in report['author_network']:
            graph = report['author_network']['graph']
            report['author_network']['graph_edges'] = list(graph.edges(data=True))
            report['author_network']['graph_nodes'] = list(graph.nodes())
            del report['author_network']['graph']
        
        # Convert sets to lists for JSON serialization
        def convert_sets_to_lists(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_sets_to_lists(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets_to_lists(item) for item in obj]
            else:
                return obj
        
        # Convert all sets to lists
        report = convert_sets_to_lists(report)
        
        # Convert Counter objects to regular dictionaries
        def convert_counters_to_dicts(obj):
            if isinstance(obj, Counter):
                return dict(obj)
            elif isinstance(obj, dict):
                return {k: convert_counters_to_dicts(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_counters_to_dicts(item) for item in obj]
            else:
                return obj
        
        report = convert_counters_to_dicts(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return output_path

def main():
    """Example usage of the AdvancedAnalytics module."""
    # Load papers
    papers_path = Path("results/all_papers_results.json")
    if not papers_path.exists():
        print("No papers data found. Please run enrichment first.")
        return
    
    with open(papers_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # Initialize analytics
    analytics = AdvancedAnalytics(papers)
    
    # Generate and save report
    output_path = analytics.save_analytics_report()
    print(f"Advanced analytics report saved to: {output_path}")
    
    # Print summary
    report = analytics.generate_comprehensive_report()
    summary = report['summary']
    
    print(f"\n📊 Analytics Summary:")
    print(f"Total Papers: {summary['total_papers']}")
    print(f"Papers with Metadata: {summary['papers_with_metadata']}")
    print(f"Papers with Authors: {summary['papers_with_authors']}")
    print(f"Papers with Affiliations: {summary['papers_with_affiliations']}")
    print(f"Papers with Sections: {summary['papers_with_sections']}")
    print(f"Papers with References: {summary['papers_with_references']}")
    print(f"Papers with DOIs: {summary['papers_with_dois']}")
    print(f"Papers with URLs: {summary['papers_with_urls']}")

if __name__ == "__main__":
    main() 