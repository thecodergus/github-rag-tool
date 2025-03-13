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
        """Cria chunks de texto a partir dos dados carregados"""
        documents = []
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Processar issues
        if self.issues_df is not None and not self.issues_df.empty:
            for _, row in self.issues_df.iterrows():
                issue_text = (
                    f"ISSUE #{row['number']}: {row['title']}\n\n{row['body'] or ''}"
                )
                chunks = text_splitter.split_text(issue_text)

                for chunk in chunks:
                    documents.append(
                        {
                            "text": chunk,
                            "metadata": {
                                "source": "issue",
                                "issue_number": row["number"],
                                "url": row["html_url"],
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
