from typing import Optional, Dict, Any, List, Union
from langchain.memory import ConversationBufferMemory, MongoDBChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import logging
import json
import os
from datetime import datetime


class ConversationManager:
    """
    Gerencia a conversação baseada em RAG (Retrieval Augmented Generation) para consultas sobre
    repositórios GitHub, integrando recuperação de documentos com memória de conversação.
    """

    def __init__(
        self,
        retriever: Any,
        model_name: str = "gpt-4o",
        session_id: Optional[str] = None,
        temperature: float = 0.7,
        memory_enabled: bool = True,
        memory_window: int = 5,
        retriever_k: int = 5,
        streaming: bool = False,
        verbose: bool = False,
    ):
        """
        Inicializa o gerenciador de conversação.

        Args:
            retriever: O recuperador de documentos a ser usado para RAG
            model_name: Nome do modelo LLM a ser utilizado
            session_id: Identificador único da sessão de conversação
            temperature: Temperatura para geração de respostas (0.0-1.0)
            memory_enabled: Se a memória de conversação deve ser habilitada
            memory_window: Número de trocas de mensagens a manter na memória
            retriever_k: Número de documentos a recuperar por consulta
            streaming: Se as respostas devem ser transmitidas em tempo real
            verbose: Se logs detalhados devem ser exibidos
        """
        self.retriever = retriever
        self.model_name = model_name
        self.session_id = (
            session_id or f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        self.temperature = temperature
        self.memory_enabled = memory_enabled
        self.memory_window = memory_window
        self.retriever_k = retriever_k
        self.streaming = streaming
        self.verbose = verbose
        self.conversation_chain = None
        self.memory = None
        self.stats = {
            "queries": 0,
            "tokens_used": 0,
            "start_time": datetime.now().isoformat(),
        }

        # Configurar logger
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)

        self.setup_conversation()

    def setup_conversation(self):
        """
        Configura a cadeia de conversação RAG com os componentes necessários:
        - Memória de conversação (local ou MongoDB)
        - LLM com configurações personalizadas
        - Cadeia de recuperação conversacional
        """
        # Configurar memória apenas se habilitada
        if self.memory_enabled:
            if self.session_id.startswith("mongodb:"):
                # Usar MongoDB como armazenamento se especificado
                mongo_connection_string = self.session_id.replace("mongodb:", "")
                self.logger.info(
                    f"Usando MongoDB para armazenamento de histórico: {self.session_id}"
                )

                message_history = MongoDBChatMessageHistory(
                    connection_string=mongo_connection_string,
                    session_id=self.session_id,
                )

                self.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer",
                    chat_memory=message_history,
                    k=self.memory_window,
                )
            else:
                # Usar memória de buffer padrão
                self.logger.info(
                    f"Usando memória de buffer local para sessão: {self.session_id}"
                )
                self.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer",
                    k=self.memory_window,
                )
        else:
            self.logger.info("Memória de conversação desabilitada")
            self.memory = None

        # Configurar callbacks para streaming se necessário
        callbacks = None
        if self.streaming:
            callbacks = CallbackManager([StreamingStdOutCallbackHandler()])

        # Template personalizado para contexto de código
        qa_prompt = PromptTemplate.from_template(
            """Você é um assistente especializado em desenvolvimento de software e código.
            Utilize as informações dos documentos para responder à pergunta do usuário.
            Se a resposta não puder ser derivada dos documentos, indique isso claramente e caso puder, dê ao usuario a fonte da informação.
            Ao discutir código, forneça explicações claras e, quando apropriado, exemplos de uso.
            
            Contexto: {context}
            
            Pergunta: {question}
            
            Resposta:"""
        )

        # Configurar LLM
        llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            streaming=self.streaming,
            callbacks=callbacks,
            verbose=self.verbose,
        )

        # Configurar cadeia de conversação
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=self.verbose,
            get_chat_history=self._get_formatted_chat_history,
        )

        self.logger.info(
            f"Cadeia de conversação configurada com modelo {self.model_name}"
        )

    def _get_formatted_chat_history(self, chat_history):
        """Formata o histórico de chat para contexto adequado"""
        formatted_history = ""
        for message in chat_history:
            if isinstance(message, tuple) and len(message) == 2:
                human, ai = message
                formatted_history += f"Humano: {human}\nAssistente: {ai}\n\n"
        return formatted_history

    def query(self, question: str) -> Dict[str, Any]:
        """
        Realiza uma consulta ao sistema RAG.

        Args:
            question: A pergunta do usuário

        Returns:
            Dicionário contendo a resposta e informações sobre as fontes utilizadas
        """
        if not self.conversation_chain:
            raise ValueError("Conversation chain não foi inicializada")

        self.logger.info(f"Processando pergunta: {question}")
        self.stats["queries"] += 1

        try:
            # Realizar a consulta
            result = self.conversation_chain({"question": question})

            # Atualizar estatísticas (estimativa simplificada de tokens)
            token_estimate = len(question) // 4 + len(result.get("answer", "")) // 4
            self.stats["tokens_used"] += token_estimate

            # Processar e formatar fontes
            sources = self._process_source_documents(result.get("source_documents", []))

            return {
                "resposta": result.get("answer", ""),
                "fontes": sources,
                "confiança": self._calculate_confidence(sources),
            }

        except Exception as e:
            self.logger.error(f"Erro ao processar consulta: {str(e)}")
            return {
                "resposta": f"Ocorreu um erro ao processar sua consulta: {str(e)}",
                "fontes": [],
                "confiança": 0.0,
            }

    def _process_source_documents(self, documents) -> List[Dict[str, Any]]:
        """
        Processa e formata os documentos-fonte recuperados.
        """
        sources = []
        for doc in documents:
            if not hasattr(doc, "metadata") or not isinstance(doc.metadata, dict):
                continue

            source_type = doc.metadata.get("source", "desconhecido")
            source_info = {
                "tipo": source_type.capitalize(),
                "relevância": (
                    doc.metadata.get("score", 0.0) if "score" in doc.metadata else None
                ),
                "conteúdo_parcial": (
                    doc.page_content[:150] + "..."
                    if len(doc.page_content) > 150
                    else doc.page_content
                ),
            }

            # Adicionar metadados específicos por tipo
            if source_type == "issue":
                source_info.update(
                    {
                        "número": doc.metadata.get("issue_number"),
                        "título": doc.metadata.get("title"),
                        "status": doc.metadata.get("state"),
                        "url": doc.metadata.get("url"),
                    }
                )
            elif source_type == "code":
                source_info.update(
                    {
                        "arquivo": doc.metadata.get("filename"),
                        "linguagem": doc.metadata.get("language", "desconhecida"),
                        "caminho": doc.metadata.get("filepath", ""),
                        "url": doc.metadata.get("url"),
                    }
                )
            elif source_type == "pull_request":
                source_info.update(
                    {
                        "número": doc.metadata.get("pr_number"),
                        "título": doc.metadata.get("title"),
                        "status": doc.metadata.get("state"),
                        "url": doc.metadata.get("url"),
                    }
                )

            sources.append(source_info)

        return sources

    def _calculate_confidence(self, sources):
        """Calcula um valor de confiança baseado nas fontes recuperadas"""
        if not sources:
            return 0.0

        # Heurística simples baseada no número e relevância das fontes
        relevance_scores = [
            s.get("relevância", 0.0) for s in sources if s.get("relevância") is not None
        ]
        avg_relevance = (
            sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
        )

        # Ajustar com base na quantidade de fontes (mais fontes = maior confiança)
        source_factor = min(1.0, len(sources) / self.retriever_k)

        return round(avg_relevance * source_factor, 2)

    def clear_memory(self):
        """Limpa a memória de conversação"""
        if self.memory_enabled and self.memory:
            self.memory.clear()
            self.logger.info("Memória de conversação limpa")
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da sessão atual"""
        self.stats["duration"] = (
            datetime.now() - datetime.fromisoformat(self.stats["start_time"])
        ).total_seconds()
        return self.stats

    def save_session(self, filepath: Optional[str] = None) -> str:
        """
        Salva o estado da sessão atual em um arquivo JSON.

        Args:
            filepath: Caminho para salvar o arquivo (opcional)

        Returns:
            Caminho do arquivo salvo
        """
        if not filepath:
            os.makedirs("./sessions", exist_ok=True)
            filepath = f"./sessions/session_{self.session_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"

        session_data = {
            "session_id": self.session_id,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "memory_enabled": self.memory_enabled,
            "stats": self.get_stats(),
            "timestamp": datetime.now().isoformat(),
        }

        try:
            with open(filepath, "w") as f:
                json.dump(session_data, f, indent=2)
            self.logger.info(f"Sessão salva em: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Erro ao salvar sessão: {str(e)}")
            return ""
