import os
import time
import json
from typing import Optional, Dict, Any

from github_rag.rag_tool.config_manager import ConfigManager
from github_rag.clients.github_client import GitHubClient
from github_rag.data_loaders.data_loader import GitHubDataLoader
from github_rag.vector_store.vector_persistence import VectorPersistence
from github_rag.managers.conversation import ConversationManager

class KnowledgeBaseBuilder:
    """
    Responsável por orquestrar a construção ou recarga da base de conhecimento.
    """

    def __init__(
        self,
        repo_url: str,
        config_manager: ConfigManager,
        github_client: Optional[GitHubClient] = None,
        persist_directory: str = "./github_rag_db",
        embeddings_model: Optional[Any] = None,
    ):
        self.repo_url = repo_url
        self.config = config_manager
        self.github_client = github_client or GitHubClient(
            repo_url, token=os.environ.get("GITHUB_API_TOKEN")
        )
        self.data_loader = GitHubDataLoader(self.github_client)
        self.vector_store = VectorPersistence(
            embeddings_model=embeddings_model,
            persist_directory=persist_directory,
            collection_name=f"github_{self._get_repo_name()}",
        )

    def _get_repo_name(self) -> str:
        parts = self.repo_url.rstrip("/").split("/")
        return f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else "unknown_repo"

    def build(
        self,
        limit_issues: Optional[int] = None,
        max_files: Optional[int] = None,
        rebuild: bool = False,
    ) -> bool:
        """
        Constrói a base de conhecimento de acordo com as configurações.
        """
        cfg = self.config.get()
        chunk_size = cfg.get("chunk_size")
        chunk_overlap = cfg.get("chunk_overlap")

        # Checar base existente
        exists = os.path.exists(self.vector_store.persist_directory) and os.listdir(self.vector_store.persist_directory)
        if exists and not rebuild:
            return self._load_existing()

        # Carregar dados
        print(f"📥 Carregando dados: issues até {limit_issues}, arquivos até {max_files}")
        self.data_loader.load_data(
            content_types=cfg.get("content_types", ["code", "issue"]),
            limit_issues=limit_issues,
            max_files=max_files,
        )

        documents = self.data_loader.create_text_chunks(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        if not documents:
            print("⚠️ Nenhum documento processado")
            return False

        print(f"🔢 Vetorizando {len(documents)} documentos...")
        success = self.vector_store.create_vector_db(documents=documents, show_progress=True)
        if not success:
            print("❌ Falha ao criar base vetorial")
            return False

        return True

    def _load_existing(self) -> bool:
        try:
            loaded = self.vector_store.load_vector_db()
            if not loaded:
                return False
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar base existente: {e}")
            return False