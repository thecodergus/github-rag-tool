from typing import List, Dict, Any, Optional
from langchain.schema.retriever import BaseRetriever

class VectorRetrieval:
    """
    Gerencia a consulta da base de vetores para sistemas RAG.
    """
    def __init__(self, vector_db: Any):
        self.vector_db = vector_db

    def get_retriever(
        self,
        search_kwargs: Optional[Dict[str, Any]] = None,
        search_type: str = "mmr",
        filter: Optional[Dict[str, Any]] = None,
    ) -> BaseRetriever:
        """
        Obtém o retriever configurado a partir da base de vetores.
        """
        if self.vector_db is None:
            raise ValueError(
                "Vector database não foi inicializado. Use VectorPersistence para criar/carregar a base primeiro."
            )

        # Configurações padrão
        if search_type == "mmr":
            default = {"k": 10, "fetch_k": 30, "lambda_mult": 0.7}
        elif search_type == "similarity_score_threshold":
            default = {"k": 10, "score_threshold": 0.75}
        else:
            default = {"k": 7}

        if search_kwargs:
            default.update(search_kwargs)
        if filter:
            default["filter"] = filter

        return self.vector_db.as_retriever(
            search_type=search_type, search_kwargs=default
        )

    def query(
        self,
        query_text: str,
        limit: int = 5,
        fetch_k: int = 20,
        include_text: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Realiza uma consulta direta na base de vetores.
        """
        if self.vector_db is None:
            raise ValueError(
                "Vector database não foi inicializado. Use VectorPersistence para criar/carregar a base primeiro."
            )

        results = self.vector_db.similarity_search_with_score(
            query=query_text, k=limit, fetch_k=fetch_k
        )
        formatted = []
        for doc, score in results:
            item = {"metadata": doc.metadata, "score": float(score)}
            if include_text:
                item["text"] = doc.page_content
            formatted.append(item)
        return formatted