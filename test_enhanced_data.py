#!/usr/bin/env python3
"""Test script to verify enhanced JSON data."""

import json
from pathlib import Path

def test_enhanced_data():
    """Test the enhanced JSON data."""
    print("🧪 Testing Enhanced JSON Data")
    print("=" * 50)
    
    # Load papers
    results_path = Path("results/all_papers_results.json")
    if not results_path.exists():
        print("❌ No results file found.")
        return
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        print(f"✅ Loaded {len(papers)} papers")
        
        # Check metadata extraction
        titles_extracted = sum(1 for p in papers if p.get("metadata", {}).get("title"))
        authors_extracted = sum(1 for p in papers if p.get("metadata", {}).get("authors"))
        keywords_extracted = sum(1 for p in papers if p.get("metadata", {}).get("keywords"))
        terms_extracted = sum(1 for p in papers if p.get("metadata", {}).get("technical_terms"))
        
        print(f"\n📊 Extraction Statistics:")
        print(f"  Papers with titles: {titles_extracted}/{len(papers)}")
        print(f"  Papers with authors: {authors_extracted}/{len(papers)}")
        print(f"  Papers with keywords: {keywords_extracted}/{len(papers)}")
        print(f"  Papers with technical terms: {terms_extracted}/{len(papers)}")
        
        # Show sample metadata
        print(f"\n📄 Sample Metadata (First Paper):")
        first_paper = papers[0]
        metadata = first_paper.get("metadata", {})
        
        print(f"  Title: {metadata.get('title', 'N/A')[:100]}...")
        print(f"  Authors: {metadata.get('authors', [])}")
        print(f"  Date: {metadata.get('date', 'N/A')}")
        print(f"  Keywords: {metadata.get('keywords', [])}")
        print(f"  Technical Terms: {metadata.get('technical_terms', [])[:5]}...")
        
        print(f"\n✅ Enhanced data verification completed!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_enhanced_data() 