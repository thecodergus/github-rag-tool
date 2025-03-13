import os
import pandas as pd
from typing import List, Dict, Optional, Any
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from github_rag.github_client import GitHubClient


class GitHubDataLoader:
    """Carregador de dados do GitHub para vectorização"""

    def __init__(self, github_client: GitHubClient):
        self.github_client = github_client
        self.issues_df = None
        self.code_files = None

    def load_data(
        self,
        content_types: List[str],
        limit_issues: Optional[int] = None,
        max_files: Optional[int] = None,
    ):
        """Carrega dados do GitHub com base nos tipos de conteúdo especificados"""
        if "issue" in content_types:
            print("🔍 Buscando issues...")
            self.issues_df = self.github_client.fetch_issues(limit=limit_issues)
            print(f"✅ Encontrados {len(self.issues_df)} issues")

        if "code" in content_types:
            print("🔍 Buscando arquivos de código...")
            self.code_files = self.github_client.fetch_code_files(max_files=max_files)
            print(f"✅ Encontrados {len(self.code_files)} arquivos de código")

    def create_text_chunks(
        self, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Cria chunks de texto a partir dos dados carregados, incluindo comentários

        Args:
            chunk_size: Tamanho de cada chunk de texto
            chunk_overlap: Quantidade de sobreposição entre chunks

        Returns:
            Lista de documentos processados
        """
        documents = []
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Processar issues e PRs com seus comentários
        if self.issues_df is not None and not self.issues_df.empty:
            for _, row in self.issues_df.iterrows():
                # Verificar se é Issue ou PR
                item_type = row.get(
                    "type", "Issue"
                )  # Se não existir, assume Issue por padrão
                item_number = row["number"]

                # Texto principal do item (Issue ou PR)
                if item_type == "Issue":
                    item_text = (
                        f"ISSUE #{item_number}: {row['title']}\n\n{row['body'] or ''}"
                    )
                    source_type = "issue"
                else:  # É um PR
                    item_text = f"PULL REQUEST #{item_number}: {row['title']}\n\n{row['body'] or ''}"
                    # Adicionar informações específicas de PR, se disponíveis
                    if "additions" in row and "deletions" in row:
                        item_text += f"\nAdições: {row.get('additions', 0)}, Exclusões: {row.get('deletions', 0)}"
                    if "merged" in row:
                        item_text += f"\nStatus de Merge: {'Mesclado' if row.get('merged', False) else 'Não mesclado'}"
                    source_type = "pull_request"

                # Adicionar comentários ao texto, se existirem
                comments = row.get("comments_data", [])
                if comments and len(comments) > 0:
                    item_text += "\n\n--- COMENTÁRIOS ---\n"
                    for i, comment in enumerate(comments, 1):
                        user = comment.get("user", "Usuário")
                        body = comment.get("body", "")
                        created_at = comment.get("created_at", "")

                        item_text += (
                            f"\nCOMENTÁRIO #{i} por {user} em {created_at}:\n{body}\n"
                        )

                # Dividir em chunks
                chunks = text_splitter.split_text(item_text)

                for chunk in chunks:
                    documents.append(
                        {
                            "text": chunk,
                            "metadata": {
                                "source": source_type,
                                "number": item_number,
                                "url": row["html_url"],
                                "title": row["title"],
                                "has_comments": len(comments) > 0,
                                "type": item_type,  # Adicionar o tipo explicitamente nos metadados
                            },
                        }
                    )

        # Processar arquivos de código (já com conteúdo)
        if self.code_files:
            for file_info in self.code_files:
                content = file_info.get("content", "")
                if content:
                    chunks = text_splitter.split_text(content)

                    for chunk in chunks:
                        documents.append(
                            {
                                "text": chunk,
                                "metadata": {
                                    "source": "code",
                                    "filename": file_info["name"],
                                    "url": file_info["url"],
                                },
                            }
                        )

        return documents
