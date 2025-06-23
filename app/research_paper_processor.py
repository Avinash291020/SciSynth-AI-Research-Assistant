"""Enhanced research paper processing for multiple papers."""
from typing import Dict, List, Set, Optional
import fitz
import re
from pathlib import Path
from datetime import datetime
import logging
from sentence_transformers import SentenceTransformer, util
from .model_cache import ModelCache

logger = logging.getLogger(__name__)

class ResearchPaperProcessor:
    def __init__(self):
        self.model = ModelCache.get_sentence_transformer()
        
    def extract_metadata(self, pdf_path: str) -> Dict:
        """Extract comprehensive metadata from PDF."""
        doc = fitz.open(pdf_path)
        metadata = {
            "title": "",
            "authors": [],
            "date": "",
            "references": [],
            "keywords": [],
            "sections": []
        }
        
        try:
            # Extract document info
            info = doc.metadata
            if info:
                metadata["title"] = info.get("title", "")
                metadata["date"] = info.get("creationDate", "")
            
            # Process each page
            text = ""
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()  # type: ignore
                text += page_text + "\n\n"
                
                # Extract title from first page if not found in metadata
                if page_num == 0 and not metadata["title"]:
                    lines = page_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        # Look for title patterns (longer lines, all caps, or title case)
                        if (len(line) > 10 and len(line) < 200 and 
                            (line.isupper() or 
                             (line[0].isupper() and line.count(' ') > 2 and 
                              not any(word.lower() in ['abstract', 'introduction', 'conclusion', 'references'] for word in line.split())))):
                            metadata["title"] = line
                            break
                
                # Look for keywords section
                if "keywords:" in page_text.lower():
                    kw_section = page_text.lower().split("keywords:")[1].split("\n")[0]
                    metadata["keywords"] = [k.strip() for k in kw_section.split(",")]
                
                # Extract section headers
                lines = page_text.split("\n")
                for line in lines:
                    if re.match(r'^\d+\.\s+[A-Z]', line):  # Numbered sections
                        metadata["sections"].append(line.strip())
            
            # Extract references from last pages
            last_pages = min(3, doc.page_count)
            for i in range(doc.page_count - last_pages, doc.page_count):
                page_text = doc[i].get_text()  # type: ignore
                if any(ref in page_text.lower() for ref in ["references", "bibliography"]):
                    refs = page_text.split("\n")
                    metadata["references"] = [r.strip() for r in refs if r.strip() 
                                           and not r.lower().startswith(("references", "bibliography"))]
            
            # Extract technical terms
            metadata["technical_terms"] = list(self.extract_technical_terms(text))
            
        finally:
            doc.close()
        
        return metadata
    
    def extract_technical_terms(self, text: str) -> Set[str]:
        """Extract technical terms and abbreviations."""
        terms = set()
        
        # Find capitalized abbreviations with definitions
        abbrev_pattern = r'\b([A-Z]{2,})\b(?:\s*\(([^)]+)\))?'
        matches = re.finditer(abbrev_pattern, text)
        for match in matches:
            abbr = match.group(1)
            definition = match.group(2)
            if definition:
                terms.add(f"{abbr} ({definition})")
            else:
                terms.add(abbr)
        
        # Find technical terms in parentheses
        tech_pattern = r'\(([^)]+(?:algorithm|model|framework|method|technique|system|approach|architecture|protocol|standard)[^)]*)\)'
        matches = re.finditer(tech_pattern, text)
        terms.update(match.group(1) for match in matches)
        
        # Find algorithm names (e.g., "Algorithm 1", "Algorithm X")
        algo_pattern = r'Algorithm\s+\d+[:\s]*([A-Za-z\s]+)'
        matches = re.finditer(algo_pattern, text)
        terms.update(match.group(1).strip() for match in matches)
        
        # Find model names (e.g., "Model X", "X Model")
        model_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Model\b'
        matches = re.finditer(model_pattern, text)
        terms.update(match.group(1) for match in matches)
        
        # Find framework names
        framework_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Framework\b'
        matches = re.finditer(framework_pattern, text)
        terms.update(match.group(1) for match in matches)
        
        # Find scientific terms (words with numbers or special characters)
        scientific_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\d*[A-Za-z]*)\b'
        matches = re.finditer(scientific_pattern, text)
        for match in matches:
            term = match.group(1).strip()
            if len(term) > 3 and any(c.isdigit() for c in term):
                terms.add(term)
        
        return terms
    
    def process_paper(self, pdf_path: Path, max_retries: int = 3) -> Optional[Dict]:
        """Process a single research paper with retries and memory management."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Processing {pdf_path.name} (attempt {attempt + 1})")
                
                # Extract metadata first
                metadata = self.extract_metadata(str(pdf_path))
                logger.info(f"Extracted metadata from {pdf_path.name}")
                
                # Process text in chunks to manage memory
                text = ""
                doc = fitz.open(str(pdf_path))
                try:
                    for page in doc:
                        text += page.get_text() + "\n\n"  # type: ignore
                        # Clear page to free memory
                        page = None
                finally:
                    doc.close()
                
                # Process the text
                from .ingest_paper import chunk_and_index
                from .insight_agent import generate_insights
                from .hypothesis_gen import generate_hypotheses
                
                chunks = chunk_and_index(text)
                insights = generate_insights(chunks)
                hypotheses = generate_hypotheses(insights)
                
                result = {
                    "paper_name": pdf_path.name,
                    "metadata": metadata,
                    "processed_date": datetime.now().isoformat(),
                    "insights": insights,
                    "hypotheses": hypotheses,
                    "num_chunks": len(chunks),
                    "text_length": len(text)
                }
                
                logger.info(f"Successfully processed {pdf_path.name}")
                return result
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to process {pdf_path.name} after {max_retries} attempts")
                    logger.error(str(e))
                    return None
                logger.warning(f"Retry {attempt + 1} for {pdf_path.name}")
                continue
    
    def analyze_cross_references(self, papers: List[Dict]) -> Dict:
        """Analyze cross-references between papers."""
        cross_refs = {
            "direct_citations": [],  # Paper A cites Paper B
            "shared_topics": [],     # Papers share similar topics
            "technical_overlap": [], # Papers use same technical terms
            "methodology_links": []  # Papers use similar methods
        }
        
        for i, paper1 in enumerate(papers):
            for paper2 in papers[i+1:]:
                # Check direct citations
                if any(paper2["paper_name"] in ref for ref in paper1["metadata"]["references"]):
                    cross_refs["direct_citations"].append({
                        "from": paper1["paper_name"],
                        "to": paper2["paper_name"]
                    })
                
                # Check shared topics
                topics1 = set(paper1["metadata"]["keywords"])
                topics2 = set(paper2["metadata"]["keywords"])
                shared = topics1.intersection(topics2)
                if shared:
                    cross_refs["shared_topics"].append({
                        "papers": [paper1["paper_name"], paper2["paper_name"]],
                        "shared_topics": list(shared)
                    })
                
                # Check technical term overlap
                terms1 = set(paper1["metadata"]["technical_terms"])
                terms2 = set(paper2["metadata"]["technical_terms"])
                shared_terms = terms1.intersection(terms2)
                if shared_terms:
                    cross_refs["technical_overlap"].append({
                        "papers": [paper1["paper_name"], paper2["paper_name"]],
                        "shared_terms": list(shared_terms)
                    })
                
                # Check methodology similarity
                text1 = " ".join(paper1["metadata"]["sections"])
                text2 = " ".join(paper2["metadata"]["sections"])
                similarity = self.compute_text_similarity(text1, text2)
                if similarity > 0.7:  # High methodology similarity
                    cross_refs["methodology_links"].append({
                        "papers": [paper1["paper_name"], paper2["paper_name"]],
                        "similarity": similarity
                    })
        
        return cross_refs
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        embeddings1 = self.model.encode([text1], convert_to_tensor=True)
        embeddings2 = self.model.encode([text2], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
        return float(similarity[0][0]) 