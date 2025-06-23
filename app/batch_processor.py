"""Batch processor for multiple research papers."""
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime
import logging
import traceback
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

from .research_paper_processor import ResearchPaperProcessor
from .model_cache import ModelCache
from .citation_network import CitationNetwork

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, papers_dir: str = "PDF Research Papers", max_workers: int | None = None):
        self.papers_dir = Path(papers_dir)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers if max_workers is not None else min(8, (os.cpu_count() or 1))
        self.processor = ResearchPaperProcessor()
        
        logger.info(f"Initialized BatchProcessor with papers directory: {self.papers_dir}")
        logger.info(f"Results will be saved to: {self.results_dir}")
        logger.info(f"Using {self.max_workers} workers for parallel processing")
        
    def process_all_papers(self) -> List[Dict]:
        """Process all PDF papers in the directory using parallel processing."""
        results = []
        pdf_files = list(self.papers_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Initialize model cache
        ModelCache()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all papers for processing
            future_to_pdf = {
                executor.submit(self.processor.process_paper, pdf_file): pdf_file 
                for pdf_file in pdf_files
            }
            
            # Process results as they complete with progress bar
            with tqdm(total=len(pdf_files), desc="Processing papers") as pbar:
                for future in as_completed(future_to_pdf):
                    pdf_file = future_to_pdf[future]
                    try:
                        result = future.result()
                        if result:  # Only add successful results
                            results.append(result)
                            
                            # Save individual paper results
                            output_file = self.results_dir / f"{pdf_file.stem}_results.json"
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                            logger.info(f"Saved results for {pdf_file.name}")
                            
                    except Exception as e:
                        logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                    finally:
                        pbar.update(1)
        
        if results:
            # Save combined results
            with open(self.results_dir / "all_papers_results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved combined results for {len(results)} papers")
            
            # Analyze cross-references
            logger.info("Analyzing cross-references between papers...")
            cross_refs = self.processor.analyze_cross_references(results)
            
            with open(self.results_dir / "cross_references.json", 'w', encoding='utf-8') as f:
                json.dump(cross_refs, f, indent=2, ensure_ascii=False)
            logger.info("Saved cross-reference analysis")
            
            # Analyze citation network
            logger.info("Analyzing citation network and paper relationships...")
            network = CitationNetwork()
            graph, relationships = network.analyze_papers(self.results_dir)
            
            # Save network analysis results
            network_results = {
                "num_papers": len(results),
                "num_citations": len([n for n in graph.nodes() if n.startswith("Citation:")]),
                "num_relationships": len(relationships),
                "relationships": relationships,
                "cross_references": cross_refs
            }
            
            with open(self.results_dir / "network_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(network_results, f, indent=2, ensure_ascii=False)
            
            # Generate visualization
            network.visualize_network(str(self.results_dir / "citation_network.png"))
            logger.info("Citation network analysis complete")
        else:
            logger.warning("No papers were successfully processed")
            
        return results

if __name__ == "__main__":
    processor = BatchProcessor()
    results = processor.process_all_papers()
    logger.info(f"Processed {len(results)} papers successfully.") 