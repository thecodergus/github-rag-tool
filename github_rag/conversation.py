from typing import Optional, Dict, Any
from langchain.memory import ConversationBufferMemory, MongoDBChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


class ConversationManager:
    """Gerencia a conversa RAG"""

    def __init__(
        self,
        retriever: Any,
        model_name: str = "gpt-3.5-turbo",
        session_id: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.retriever = retriever
        self.model_name = model_name
        self.session_id = session_id or "default"
        self.temperature = temperature
        self.conversation_chain = None
        self.setup_conversation()

    def setup_conversation(self):
        """Configura a cadeia de conversação"""
        # Configurar memória
        if self.session_id.startswith("mongodb:"):
            # Usar MongoDB como armazenamento se especificado
            mongo_connection_string = self.session_id.replace("mongodb:", "")
            message_history = MongoDBChatMessageHistory(
                connection_string=mongo_connection_string, session_id=self.session_id
            )
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                chat_memory=message_history,
            )
        else:
            # Usar memória de buffer padrão
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key="answer"
            )

        # Configurar LLM
        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        # Configurar cadeia de conversação
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.retriever,
            memory=memory,
            return_source_documents=True,
        )

    def query(self, question: str) -> Dict[str, Any]:
        """Realiza uma consulta ao sistema RAG"""
        if not self.conversation_chain:
            raise ValueError("Conversation chain não foi inicializada")

        print(f"🤔 Processando pergunta: {question}")
        result = self.conversation_chain({"question": question})

        # Formatar fontes
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                if doc.metadata["source"] == "issue":
                    sources.append(
                        {
                            "tipo": "Issue",
                            "número": doc.metadata["issue_number"],
                            "url": doc.metadata["url"],
                        }
                    )
                elif doc.metadata["source"] == "code":
                    sources.append(
                        {
                            "tipo": "Código",
                            "arquivo": doc.metadata["filename"],
                            "url": doc.metadata["url"],
                        }
                    )

        return {"resposta": result["answer"], "fontes": sources}
