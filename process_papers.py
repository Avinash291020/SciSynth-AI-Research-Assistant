#!/usr/bin/env python3
"""Script to process research papers with detailed logging."""
import logging
import sys
from app.batch_processor import BatchProcessor

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Force output to stdout
)

# Disable other loggers
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

def main():
    logger = logging.getLogger('paper_processor')
    logger.info("Starting paper processing...")
    
    try:
        processor = BatchProcessor()
        results = processor.process_all_papers()
        logger.info(f"Successfully processed {len(results)} papers")
        
    except Exception as e:
        logger.error("Failed to process papers:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 