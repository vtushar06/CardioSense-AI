import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorStoreManager:
    """Manages the FAISS vector store for medical knowledge retrieval."""

    def __init__(
        self, 
        index_path: str = "models/vector_store", 
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store: Optional[FAISS] = None
        
        # Load if exists
        self.load_index()

    def create_index(self, documents: List[Document]):
        """Creates a new FAISS index from a list of documents."""
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.save_index()

    def add_documents(self, documents: List[Document]):
        """Appends documents to the existing index."""
        if self.vector_store is None:
            self.create_index(documents)
        else:
            self.vector_store.add_documents(documents)
            self.save_index()

    def save_index(self):
        """Persists the FAISS index to disk."""
        if self.vector_store:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            self.vector_store.save_local(self.index_path)

    def load_index(self):
        """Loads a FAISS index from disk if it exists."""
        if os.path.exists(self.index_path):
            self.vector_store = FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True  # Local trusted store
            )

    def search(self, query: str, k: int = 3) -> List[Document]:
        """Performs similarity search against the index."""
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)

# Simple test script
if __name__ == "__main__":
    manager = VectorStoreManager()
    # docs = [Document(page_content="Cardiovascular disease is the leading cause of death.")]
    # manager.create_index(docs)
    # results = manager.search("What is the leading cause of death?")
    # print(results)
