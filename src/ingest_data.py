import sys
import os
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Add the project root to the sys.path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vector_store import VectorStoreManager

def ingest_guidelines(file_path: str = "data/cardio_guidelines.md"):
    """Reads the guidelines file, splits it by headers, and indexes it into FAISS."""
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, "r") as f:
        markdown_content = f.read()

    # Split by headers to maintain context for each section
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = splitter.split_text(markdown_content)
    
    print(f"Split document into {len(chunks)} contextual chunks.")

    manager = VectorStoreManager()
    manager.create_index(chunks)
    
    print(f"Successfully indexed guidelines to {manager.index_path}")

if __name__ == "__main__":
    ingest_guidelines()
