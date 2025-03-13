import os
from typing import List, Optional, Dict, Any, Union, Tuple, Callable
import json
import time
from datetime import datetime

from github_rag.github_client import GitHubClient
from github_rag.data_loader import GitHubDataLoader
from github_rag.vector_store import VectorStore
from github_rag.conversation import ConversationManager
from github_rag.utils import generate_session_id


class GitHubRagTool:
    """
    Ferramenta RAG para GitHub com capacidades avançadas de memória e recuperação.

    Esta classe coordena todos os componentes para criar um sistema RAG
    baseado em repositórios do GitHub, permitindo consultas em linguagem natural
    sobre issues, pull requests e código fonte com contexto enriquecido.
    """

    def __init__(
        self,
        repo_url: str,
        content_types: List[str] = ["code", "issue"],
        custom_model: str = "gpt-3.5-turbo",
        session_id: Optional[str] = None,
        temperature: float = 0.7,
        persist_directory: str = "./github_rag_db",
        embeddings_model: Optional[Any] = None,
    ):
        """
        Inicializa a ferramenta RAG para GitHub.

        Args:
            repo_url: URL do repositório GitHub
            content_types: Tipos de conteúdo para indexar ("code", "issue")
            custom_model: Modelo LLM a ser usado
            session_id: ID de sessão para persistência de memória
            temperature: Temperatura para o modelo LLM
            persist_directory: Diretório para persistir a base de vetores
            embeddings_model: Modelo de embeddings personalizado (opcional)
        """
        self.repo_url = repo_url
        self.content_types = content_types
        self.model_name = custom_model
        self.session_id = session_id or generate_session_id()
        self.temperature = temperature
        self.persist_directory = persist_directory

        # Status de inicialização dos componentes
        self.is_data_loaded = False
        self.is_vectordb_ready = False
        self.is_conversation_ready = False

        # Configurações avançadas com valores padrão
        self.config = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retriever_k": 5,
            "retriever_fetch_k": 20,
            "retriever_search_type": "mmr",
            "use_memory": True,
            "memory_window": 10,
            "max_tokens": 60_000,
        }

        # Inicializar componentes básicos
        self.github_client = GitHubClient(repo_url)
        self.data_loader = GitHubDataLoader(self.github_client)
        self.vector_store = VectorStore(
            embeddings_model=embeddings_model,
            persist_directory=persist_directory,
            collection_name=f"github_{self._get_repo_name()}",
        )
        self.conversation_manager = None

        # Métricas e estatísticas
        self.stats = {
            "queries_count": 0,
            "tokens_used": 0,
            "last_query_time": None,
            "avg_response_time": 0,
            "total_response_time": 0,
        }

        print(f"🚀 GitHubRagTool inicializado para {repo_url}")
        print(f"📂 Tipos de conteúdo: {', '.join(content_types)}")
        print(f"🧠 Modelo: {custom_model}")
        print(f"🔑 Sessão: {self.session_id}")

    def _get_repo_name(self) -> str:
        """Extrai o nome do repositório da URL"""
        parts = self.repo_url.rstrip("/").split("/")
        if len(parts) >= 2:
            return f"{parts[-2]}_{parts[-1]}"
        return "unknown_repo"

    def configure(self, config_options: Dict[str, Any]) -> None:
        """
        Configura opções avançadas da ferramenta.

        Args:
            config_options: Dicionário com opções de configuração
        """
        self.config.update(config_options)
        print(f"⚙️ Configurações atualizadas: {json.dumps(self.config, indent=2)}")

        # Atualizar componentes afetados pelas novas configurações
        if self.conversation_manager and "use_memory" in config_options:
            self.conversation_manager.set_memory_enabled(self.config["use_memory"])

        if self.conversation_manager and "memory_window" in config_options:
            self.conversation_manager.set_memory_window(self.config["memory_window"])

    def build_knowledge_base(
        self,
        limit_issues: Optional[int] = None,
        max_files: Optional[int] = None,
        rebuild: bool = False,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> bool:
        """
        Constrói a base de conhecimento a partir do repositório GitHub.

        Args:
            limit_issues: Limite de issues a carregar (None para todos)
            max_files: Limite de arquivos a carregar (None para todos)
            rebuild: Se deve reconstruir a base mesmo se já existir
            chunk_size: Tamanho de cada chunk (sobrescreve config)
            chunk_overlap: Sobreposição entre chunks (sobrescreve config)

        Returns:
            True se a construção foi bem-sucedida, False caso contrário
        """
        start_time = time.time()

        # Verificar se já existe uma base que pode ser carregada
        db_exists = os.path.exists(self.persist_directory) and os.listdir(
            self.persist_directory
        )

        if db_exists and not rebuild:
            print(
                f"📁 Base de conhecimento existente encontrada em {self.persist_directory}"
            )
            success = self._load_existing_knowledge_base()
            if success:
                return True
            print("⚠️ Falha ao carregar base existente. Reconstruindo...")

        # Atualizar configurações locais se fornecidas
        if chunk_size:
            self.config["chunk_size"] = chunk_size
        if chunk_overlap:
            self.config["chunk_overlap"] = chunk_overlap

        try:
            # Carregar dados
            print(f"📥 Carregando dados do repositório {self.repo_url}...")
            self.data_loader.load_data(
                content_types=self.content_types,
                limit_issues=limit_issues,
                max_files=max_files,
            )
            self.is_data_loaded = True

            # Obter estatísticas dos dados carregados
            data_summary = self.data_loader.get_data_summary()
            print(f"📊 Dados carregados: {json.dumps(data_summary, indent=2)}")

            # Criar chunks com as configurações especificadas
            print(
                f"✂️ Criando chunks (tamanho={self.config['chunk_size']}, sobreposição={self.config['chunk_overlap']})..."
            )
            documents = self.data_loader.create_text_chunks(
                chunk_size=self.config["chunk_size"],
                chunk_overlap=self.config["chunk_overlap"],
            )

            if not documents:
                print("⚠️ Nenhum documento foi processado")
                return False

            print(f"📄 {len(documents)} chunks criados")

            # Construir a base de vetores
            print("🔢 Vetorizando documentos...")
            success = self.vector_store.create_vector_db(
                documents=documents, show_progress=True
            )

            if not success:
                print("❌ Falha ao criar base de vetores")
                return False

            self.is_vectordb_ready = True

            # Configurar o gerenciador de conversas
            self._setup_conversation_manager()

            elapsed_time = time.time() - start_time
            print(
                f"✅ Base de conhecimento construída com sucesso em {elapsed_time:.2f} segundos"
            )
            return True

        except Exception as e:
            print(f"❌ Erro ao construir base de conhecimento: {str(e)}")
            return False

    def _load_existing_knowledge_base(self) -> bool:
        """
        Carrega uma base de conhecimento existente.

        Returns:
            True se carregada com sucesso, False caso contrário
        """
        try:
            # Tentar carregar a base vetorial
            success = self.vector_store.load_vector_db()

            if not success:
                return False

            self.is_vectordb_ready = True

            # Configurar gerenciador de conversas
            self._setup_conversation_manager()

            print("✅ Base de conhecimento carregada com sucesso")
            return True

        except Exception as e:
            print(f"❌ Erro ao carregar base existente: {str(e)}")
            return False

    def _setup_conversation_manager(self) -> None:
        """Configura o gerenciador de conversas com base nas configurações atuais"""
        # Configurar o retriever com as configurações atuais
        retriever = self.vector_store.get_retriever(
            search_kwargs={
                "k": self.config["retriever_k"],
                "fetch_k": self.config["retriever_fetch_k"],
            }
        )

        # Criar o gerenciador de conversas
        self.conversation_manager = ConversationManager(
            retriever=retriever,
            model_name=self.model_name,
            session_id=self.session_id,
            temperature=self.temperature,
            memory_enabled=self.config["use_memory"],
            memory_window=self.config["memory_window"],
            max_tokens=self.config["max_tokens"],
        )

        self.is_conversation_ready = True

    def query(
        self,
        question: str,
        stream: bool = False,
        callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Realiza uma consulta ao sistema RAG.

        Args:
            question: Pergunta em linguagem natural
            stream: Se deve transmitir a resposta incrementalmente
            callback: Função de callback para streaming (recebe chunks de texto)

        Returns:
            Dicionário com resposta, fontes e metadados

        Raises:
            ValueError: Se a base de conhecimento não estiver pronta
        """
        if not self.is_conversation_ready:
            raise ValueError(
                "Base de conhecimento não está pronta. Execute build_knowledge_base() primeiro."
            )

        # Registrar métricas
        query_start_time = time.time()
        self.stats["queries_count"] += 1
        self.stats["last_query_time"] = datetime.now().isoformat()

        try:
            # Executar a consulta
            if stream and callback:
                result = self.conversation_manager.query_with_streaming(
                    question, callback
                )
            else:
                result = self.conversation_manager.query(question)

            # Processar o resultado para adicionar metadados
            processed_result = self._process_query_result(result)

            # Atualizar métricas
            query_time = time.time() - query_start_time
            self.stats["total_response_time"] += query_time
            self.stats["avg_response_time"] = (
                self.stats["total_response_time"] / self.stats["queries_count"]
            )

            # Adicionar tempo de resposta ao resultado
            processed_result["metrics"] = {
                "response_time_seconds": query_time,
                "query_index": self.stats["queries_count"],
            }

            return processed_result

        except Exception as e:
            error_result = {
                "answer": f"Erro ao processar a consulta: {str(e)}",
                "sources": [],
                "success": False,
                "error": str(e),
            }
            return error_result

    def _process_query_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa o resultado da consulta para adicionar informações úteis.

        Args:
            result: Resultado original da consulta

        Returns:
            Resultado processado com metadados adicionais
        """
        # Começamos com o resultado original
        processed = result.copy()

        # Adicionar flag de sucesso
        processed["success"] = True

        # Adicionar sessão e timestamp
        processed["session_id"] = self.session_id
        processed["timestamp"] = datetime.now().isoformat()

        # Processar fontes para destacar os trechos mais relevantes
        if "sources" in processed and processed["sources"]:
            for i, source in enumerate(processed["sources"]):
                # Limitar o tamanho do texto da fonte para economizar espaço
                if "content" in source and len(source["content"]) > 500:
                    source["content_preview"] = source["content"][:500] + "..."

                # Adicionar índice da fonte para referência mais fácil
                source["index"] = i + 1

        return processed

    def search_sources(
        self, query: str, limit: int = 5, include_content: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Busca fontes diretamente sem gerar resposta via LLM.

        Args:
            query: Texto da consulta
            limit: Número máximo de resultados
            include_content: Se deve incluir o conteúdo completo

        Returns:
            Lista de fontes relevantes com seus metadados

        Raises:
            ValueError: Se a base de conhecimento não estiver pronta
        """
        if not self.is_vectordb_ready:
            raise ValueError(
                "Base de conhecimento não está pronta. Execute build_knowledge_base() primeiro."
            )

        try:
            # Buscar diretamente na base vetorial
            search_results = self.vector_store.query(
                query_text=query, limit=limit, include_text=include_content
            )

            # Processar resultados para um formato mais amigável
            processed_results = []
            for i, result in enumerate(search_results):
                processed = {
                    "index": i + 1,
                    "score": result["score"],
                    "metadata": result["metadata"],
                }

                if include_content and "text" in result:
                    processed["content"] = result["text"]

                processed_results.append(processed)

            return processed_results

        except Exception as e:
            print(f"❌ Erro ao buscar fontes: {str(e)}")
            return []

    def add_more_content(
        self,
        content_types: List[str],
        limit_issues: Optional[int] = None,
        max_files: Optional[int] = None,
    ) -> bool:
        """
        Adiciona mais conteúdo à base de conhecimento existente.

        Args:
            content_types: Tipos de conteúdo a adicionar
            limit_issues: Limite de issues a carregar
            max_files: Limite de arquivos a carregar

        Returns:
            True se o conteúdo foi adicionado com sucesso, False caso contrário

        Raises:
            ValueError: Se a base de conhecimento não estiver inicializada
        """
        if not self.is_vectordb_ready:
            raise ValueError(
                "Base de conhecimento não está inicializada. Execute build_knowledge_base() primeiro."
            )

        try:
            # Carregar apenas os novos tipos de conteúdo
            print(f"📥 Carregando conteúdo adicional: {', '.join(content_types)}...")
            self.data_loader.load_data(
                content_types=content_types,
                limit_issues=limit_issues,
                max_files=max_files,
            )

            # Criar chunks com as configurações atuais
            documents = self.data_loader.create_text_chunks(
                chunk_size=self.config["chunk_size"],
                chunk_overlap=self.config["chunk_overlap"],
            )

            if not documents:
                print("⚠️ Nenhum documento adicional foi processado")
                return False

            print(f"📄 {len(documents)} chunks adicionais criados")

            # Adicionar à base vetorial existente
            success = self.vector_store.add_documents(documents=documents)

            if not success:
                print("❌ Falha ao adicionar novos documentos à base")
                return False

            print(
                f"✅ {len(documents)} documentos adicionados com sucesso à base existente"
            )
            return True

        except Exception as e:
            print(f"❌ Erro ao adicionar conteúdo: {str(e)}")
            return False

    def reset(self, delete_db: bool = False) -> bool:
        """
        Reseta o estado da ferramenta, opcionalmente excluindo a base de dados.

        Args:
            delete_db: Se deve excluir fisicamente a base de dados vetorial

        Returns:
            True se o reset foi bem-sucedido, False caso contrário
        """
        try:
            # Resetar estado interno
            self.is_data_loaded = False
            self.is_vectordb_ready = False
            self.is_conversation_ready = False
            self.conversation_manager = None

            # Resetar estatísticas
            self.stats = {
                "queries_count": 0,
                "tokens_used": 0,
                "last_query_time": None,
                "avg_response_time": 0,
                "total_response_time": 0,
            }

            # Excluir base de dados se solicitado
            if delete_db and self.vector_store:
                success = self.vector_store.delete_collection()
                if not success:
                    print("⚠️ Aviso: Falha ao excluir a coleção vetorial")

            print("🔄 Estado da ferramenta resetado com sucesso")
            return True

        except Exception as e:
            print(f"❌ Erro ao resetar: {str(e)}")
            return False

    def save_session(self, directory: Optional[str] = None) -> bool:
        """
        Salva o estado da sessão atual para uso futuro.

        Args:
            directory: Diretório onde salvar os dados da sessão

        Returns:
            True se a sessão foi salva com sucesso, False caso contrário
        """
        save_dir = directory or f"./sessions/{self.session_id}"
        os.makedirs(save_dir, exist_ok=True)

        try:
            # Salvar metadados e configurações
            metadata = {
                "session_id": self.session_id,
                "repo_url": self.repo_url,
                "model_name": self.model_name,
                "content_types": self.content_types,
                "config": self.config,
                "stats": self.stats,
                "created_at": datetime.now().isoformat(),
            }

            with open(f"{save_dir}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Salvar histórico de conversas se disponível
            if self.conversation_manager and hasattr(
                self.conversation_manager, "get_history"
            ):
                history = self.conversation_manager.get_history()

                with open(f"{save_dir}/conversation_history.json", "w") as f:
                    json.dump(history, f, indent=2)

            print(f"💾 Sessão salva em {save_dir}")
            return True

        except Exception as e:
            print(f"❌ Erro ao salvar sessão: {str(e)}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Retorna o status atual da ferramenta.

        Returns:
            Dicionário com informações de status
        """
        vector_db_stats = (
            self.vector_store.get_stats() if self.is_vectordb_ready else None
        )

        status = {
            "session_id": self.session_id,
            "repo_url": self.repo_url,
            "model_name": self.model_name,
            "content_types": self.content_types,
            "is_data_loaded": self.is_data_loaded,
            "is_vectordb_ready": self.is_vectordb_ready,
            "is_conversation_ready": self.is_conversation_ready,
            "config": self.config,
            "stats": self.stats,
            "vector_db": vector_db_stats,
        }

        return status
