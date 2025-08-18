import os
import time
from typing import Optional, Dict, Any

from github_rag.rag_tool.config_manager import ConfigManager
from github_rag.rag_tool.knowledge_base_builder import KnowledgeBaseBuilder
from github_rag.managers.conversation import ConversationManager
from github_rag.vector_store.vector_retrieval import VectorRetrieval

class SessionManager:
    """
    Orquestra a configuração, construção da base de conhecimento e gerenciamento de sessão RAG.
    """
    def __init__(
        self,
        repo_url: str,
        initial_config: Dict[str, Any],
        embeddings_model: Optional[Any] = None,
    ):
        self.repo_url = repo_url
        self.config_manager = ConfigManager(initial_config)
        self.kb_builder = KnowledgeBaseBuilder(
            repo_url=repo_url,
            config_manager=self.config_manager,
            embeddings_model=embeddings_model,
        )
        self.conv_manager: Optional[ConversationManager] = None

    def setup(
        self,
        limit_issues: Optional[int] = None,
        max_files: Optional[int] = None,
        rebuild: bool = False,
    ) -> bool:
        """
        Inicializa a base de conhecimento e gerador de conversação RAG.
        """
        # Construir ou recarregar a base de conhecimento
        success = self.kb_builder.build(
            limit_issues=limit_issues,
            max_files=max_files,
            rebuild=rebuild,
        )
        if not success:
            return False

        # Inicializar gerenciador de conversação com retriever configurado
        config = self.config_manager.get()
        retriever = VectorRetrieval(self.kb_builder.vector_store.vector_db).get_retriever(
            search_kwargs={"k": config.get("retriever_k", 5)},
            search_type=config.get("search_type", "mmr"),
            filter=config.get("filter"),
        )
        self.conv_manager = ConversationManager(
            retriever=retriever,
            model_name=config.get("custom_model", os.environ.get("OPENAI_MODEL")),
            session_id=config.get("session_id"),
            temperature=config.get("temperature", 0.7),
            memory_enabled=config.get("use_memory", True),
            memory_window=config.get("memory_window", 5),
            retriever_k=config.get("retriever_k", 5),
            streaming=config.get("streaming", False),
            verbose=config.get("verbose", False),
        )
        return True

    def query(self, question: str) -> Dict[str, Any]:
        """
        Executa uma consulta na sessão RAG.
        """
        if not self.conv_manager:
            raise RuntimeError("Gerenciador de conversação não inicializado. Chame setup() primeiro.")
        return self.conv_manager.query(question)

    def get_status(self) -> Dict[str, Any]:
        """
        Retorna status atual da sessão e da base.
        """
        if not self.conv_manager:
            raise RuntimeError("Gerenciador de conversação não inicializado. Chame setup() primeiro.")
        vector_stats = self.kb_builder.vector_store.get_stats()
        return {
            "session_id": self.conv_manager.session_id,
            "stats": self.conv_manager.get_stats(),
            "vector_db": vector_stats,
            "is_vectordb_ready": vector_stats.get("status") == "inicializada",
        }

    def search_sources(self, query: str, limit: int = 5) -> Any:
        """
        Busca diretamente na base vetorial.
        """
        return self.kb_builder.vector_store.query(query_text=query, limit=limit)

    def save_session(self, directory: Optional[str] = None) -> bool:
        """
        Salva dados da sessão para uso futuro.
        """
        if not self.conv_manager:
            raise RuntimeError("Gerenciador de conversação não inicializado. Chame setup() primeiro.")
        saved_path = self.conv_manager.save_session(directory)
        return bool(saved_path)