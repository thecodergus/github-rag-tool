from typing import List, Any, Dict

class SourceDocumentProcessor:
    """
    Processa documentos-fonte recuperados pela cadeia RAG.
    """
    def process(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Processa e formata os documentos-fonte recuperados.
        """
        sources: List[Dict[str, Any]] = []
        for doc in documents:
            if not hasattr(doc, "metadata") or not isinstance(doc.metadata, dict):
                continue

            source_type = doc.metadata.get("source", "desconhecido")
            source_info: Dict[str, Any] = {
                "tipo": source_type.capitalize(),
                "relevância": doc.metadata.get("score", None),
                "conteúdo_parcial": (
                    doc.page_content[:150] + "..."
                    if len(doc.page_content) > 150 else doc.page_content
                ),
            }

            if source_type == "issue":
                source_info.update({
                    "número": doc.metadata.get("issue_number"),
                    "título": doc.metadata.get("title"),
                    "status": doc.metadata.get("state"),
                    "url": doc.metadata.get("url"),
                })
            elif source_type == "code":
                source_info.update({
                    "arquivo": doc.metadata.get("filename"),
                    "linguagem": doc.metadata.get("language", "desconhecida"),
                    "caminho": doc.metadata.get("filepath", ""),
                    "url": doc.metadata.get("url"),
                })
            elif source_type == "pull_request":
                source_info.update({
                    "número": doc.metadata.get("pr_number"),
                    "título": doc.metadata.get("title"),
                    "status": doc.metadata.get("state"),
                    "url": doc.metadata.get("url"),
                })

            sources.append(source_info)
        return sources