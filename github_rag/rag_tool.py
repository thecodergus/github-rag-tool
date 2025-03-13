from typing import List, Optional, Dict, Any

from github_rag.github_client import GitHubClient
from github_rag.data_loader import GitHubDataLoader
from github_rag.vector_store import VectorStore
from github_rag.conversation import ConversationManager
from github_rag.utils import generate_session_id


class GitHubRagTool:
    """
    Ferramenta RAG para GitHub com capacidades de memória.

    Esta classe coordena todos os componentes para criar um sistema RAG
    baseado em repositórios do GitHub, permitindo consultas em linguagem natural
    sobre issues e código fonte.
    """

    def __init__(
        self,
        repo_url: str,
        content_types: List[str] = ["code", "issue"],
        custom_model: str = "gpt-3.5-turbo",
        session_id: Optional[str] = None,
        temperature: float = 0.7,
    ):
        """
        Inicializa a ferramenta RAG para GitHub.

        Args:
            repo_url: URL do repositório GitHub
            content_types: Tipos de conteúdo para indexar ("code", "issue")
            custom_model: Modelo LLM a ser usado
            session_id: ID de sessão para persistência de memória
            temperature: Temperatura para o modelo LLM
        """
        self.repo_url = repo_url
        self.content_types = content_types
        self.model_name = custom_model
        self.session_id = session_id or generate_session_id()
        self.temperature = temperature

        # Inicializar componentes
        self.github_client = GitHubClient(repo_url)
        self.data_loader = GitHubDataLoader(self.github_client)
        self.vector_store = VectorStore()
        self.conversation_manager = None

        print(f"🚀 GitHubRagTool inicializado para {repo_url}")
        print(f"📂 Tipos de conteúdo: {', '.join(content_types)}")
        print(f"🧠 Modelo: {custom_model}")
        print(f"🔑 Sessão: {self.session_id}")

    def build_knowledge_base(
        self, limit_issues: Optional[int] = None, max_files: Optional[int] = None
    ):
        """
        Constrói a base de conhecimento a partir do repositório GitHub.

        Args:
            limit_issues: Limite de issues a carregar (None para todos)
            max_files: Limite de arquivos a carregar (None para todos)
        """
        # Carregar dados - sempre passar None para max_files para não limitar
        self.data_loader.load_data(
            content_types=self.content_types,
            limit_issues=limit_issues,
            max_files=None,  # Sempre None para baixar todos os arquivos
        )

        # Criar chunks e vetorizar
        documents = self.data_loader.create_text_chunks()
        self.vector_store.create_vector_db(documents)

        # Configurar a conversa
        retriever = self.vector_store.get_retriever()
        self.conversation_manager = ConversationManager(
            retriever=retriever,
            model_name=self.model_name,
            session_id=self.session_id,
            temperature=self.temperature,
        )

        print("✅ Base de conhecimento construída com sucesso")

    def query(self, question: str) -> Dict[str, Any]:
        """
        Realiza uma consulta ao sistema RAG.

        Args:
            question: Pergunta em linguagem natural

        Returns:
            Dicionário com resposta e fontes
        """
        if not self.conversation_manager:
            raise ValueError(
                "Base de conhecimento não foi construída. Execute build_knowledge_base() primeiro."
            )

        return self.conversation_manager.query(question)
