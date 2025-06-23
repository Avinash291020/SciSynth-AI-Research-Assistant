from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from llama_index.core import VectorStoreIndex

def summarize_index(index: VectorStoreIndex):
    query_engine = index.as_query_engine()
    response = query_engine.query("Summarize the document")
    return str(response)

if __name__ == "__main__":
    from ingest_paper import chunk_and_index, extract_text_from_pdf
    index = chunk_and_index(extract_text_from_pdf("example_paper.pdf"))
    summary = summarize_index(index)
    print("🔍 Summary:\n", summary) 