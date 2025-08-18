import pandas as pd
from typing import List, Dict, Any

class IssueProcessor:
    """
    Processa dados de issues e pull requests para criar documentos de texto.
    """

    def __init__(self, text_splitter):
        self.text_splitter = text_splitter

    def process(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Processa o DataFrame de issues e PRs, retornando lista de documentos.
        """
        if df is None or df.empty:
            return []
        documents = []
        for _, row in df.iterrows():
            is_pr = "pull_request" in row
            item_type = "Pull Request" if is_pr else "Issue"
            source_type = "pull_request" if is_pr else "issue"
            item_number = row["number"]
            # Construir texto principal
            item_text = self._build_item_text(row, item_type, item_number)
            # Adicionar comentários
            item_text = self._add_comments_to_text(item_text, row)
            # Dividir em chunks
            chunks = self.text_splitter.split_text(item_text)
            # Criar metadados
            metadata = self._create_item_metadata(row, item_number, item_type)
            # Criar documentos a partir dos chunks
            documents.extend(
                self._create_documents_from_chunks(chunks, source_type, metadata)
            )
        return documents

    def _build_item_text(self, row: pd.Series, item_type: str, item_number: int) -> str:
        item_text = (
            f"{item_type.upper()} #{item_number}: {row['title']}\n\n{row['body'] or ''}"
        )
        if item_type == "Pull Request":
            if "additions" in row and "deletions" in row:
                item_text += f"\nAdições: {row.get('additions', 0)}, Exclusões: {row.get('deletions', 0)}"
            if "merged" in row:
                item_text += f"\nStatus de Merge: {'Mesclado' if row.get('merged', False) else 'Não mesclado'}"
        return item_text

    def _add_comments_to_text(self, item_text: str, row: pd.Series) -> str:
        comments = row.get("comments_data", [])
        if not comments:
            return item_text
        item_text += "\n\n--- COMENTÁRIOS ---\n"
        for i, comment in enumerate(comments, 1):
            user = comment.get("user", "Usuário")
            body = comment.get("body", "")
            created_at = comment.get("created_at", "")
            item_text += f"\nCOMENTÁRIO #{i} por {user} em {created_at}:\n{body}\n"
        return item_text

    def _create_item_metadata(self, row: pd.Series, item_number: int, item_type: str) -> Dict[str, Any]:
        comments = row.get("comments_data", [])
        return {
            "source": "pull_request" if item_type == "Pull Request" else "issue",
            "number": item_number,
            "url": row["html_url"],
            "title": row["title"],
            "has_comments": len(comments) > 0,
            "type": item_type,
        }

    def _create_documents_from_chunks(
        self, chunks: List[str], source_type: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            documents.append({"text": chunk, "metadata": chunk_metadata})
        return documents