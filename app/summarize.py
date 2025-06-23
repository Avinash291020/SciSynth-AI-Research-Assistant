try:
    from langchain.chat_models import ChatOpenAI
    from langchain.chains.summarize import load_summarize_chain
    from langchain.docstore.document import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: langchain not available. Summarization features will be limited.")

try:
    from llama_index.core import VectorStoreIndex
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    print("Warning: llama_index not available. Summarization features will be limited.")

def summarize_index(index):
    """Summarize the given index if dependencies are available."""
    if not LLAMA_INDEX_AVAILABLE:
        return "Summarization not available: llama_index not installed"
    
    try:
        query_engine = index.as_query_engine()
        response = query_engine.query("Summarize the document")
        return str(response)
    except Exception as e:
        return f"Error during summarization: {str(e)}"

if __name__ == "__main__":
    if not LLAMA_INDEX_AVAILABLE:
        print("Cannot run: llama_index not available")
    else:
        from ingest_paper import chunk_and_index, extract_text_from_pdf
        index = chunk_and_index(extract_text_from_pdf("example_paper.pdf"))
        summary = summarize_index(index)
        print("🔍 Summary:\n", summary) 