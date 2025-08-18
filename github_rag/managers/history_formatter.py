from typing import Any, List

class HistoryFormatter:
    """
    Formata histórico de conversação para uso pelo ConversationalRetrievalChain.
    """
    def format(self, chat_history: List[Any]) -> str:
        """
        Recebe uma lista de mensagens de chat e retorna uma string formatada.
        Cada mensagem pode ter atributos 'role' ou 'type' e 'content' ou 'text'.
        """
        formatted = []
        for message in chat_history:
            # Tenta ler atributos comuns
            role = getattr(message, "type", None) or getattr(message, "role", None) or "message"
            content = getattr(message, "content", None) or getattr(message, "text", None) or str(message)
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)