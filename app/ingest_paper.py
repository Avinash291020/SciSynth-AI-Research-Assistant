import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import os
from typing import Any, List
import re

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters while preserving important punctuation
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s*([.,;:!?()])\s*', r'\1 ', text)
    
    # Remove multiple newlines
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    doc: Any = fitz.open(pdf_path)
    full_text = ""
    
    try:
        for page in doc:
            text = page.get_text("text")
            # Clean each page's text
            text = clean_text(text)
            full_text += text + "\n\n"
    finally:
        doc.close()
    
    return full_text.strip()

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks of approximately equal size."""
    # Split into paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If a single paragraph is too long, split it into sentences
        if len(para) > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if current_size + len(sentence) > chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                current_chunk.append(sentence)
                current_size += len(sentence) + 1
        else:
            if current_size + len(para) > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(para)
            current_size += len(para) + 2  # +2 for paragraph break
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def chunk_and_index(text: str, save_dir: str = "data/ingested") -> List[str]:
    """Chunk the text and create embeddings."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save original text
    with open(os.path.join(save_dir, "input.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    
    # Create chunks
    chunks = chunk_text(text)
    
    return chunks

if __name__ == "__main__":
    pdf_path = "example_paper.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_and_index(text)
    print("✅ Paper ingested and chunked into", len(chunks), "segments.") 