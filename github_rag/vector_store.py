import os
from typing import List, Dict, Any, Optional
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


class VectorStore:
    """Gerencia a base de vetores para o RAG"""

    def __init__(self, embeddings_model: Optional[Any] = None):
        self.embeddings = embeddings_model or OpenAIEmbeddings()
        self.vector_db = None

    def create_vector_db(
        self,
        documents: List[Dict[str, Any]],
        persist_directory: str = "./github_rag_db",
    ):
        """Cria base de vetores a partir dos documentos"""
        if not documents:
            print("⚠️ Nenhum documento para vetorizar")
            return

        os.makedirs(persist_directory, exist_ok=True)

        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        print(f"🔢 Criando vetores para {len(texts)} documentos...")
        self.vector_db = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=self.embeddings,
            persist_directory=persist_directory,
        )
        print("✅ Base de vetores criada com sucesso")

    def get_retriever(self, search_kwargs: Dict[str, Any] = None):
        """Obtém o retriever da base de vetores"""
        if self.vector_db is None:
            raise ValueError("Vector database não foi inicializado")

        default_search_kwargs = {"k": 5}
        if search_kwargs:
            default_search_kwargs.update(search_kwargs)

        return self.vector_db.as_retriever(search_kwargs=default_search_kwargs)
