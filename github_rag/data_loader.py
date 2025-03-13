import os
import pandas as pd
from typing import List, Dict, Optional, Any, Union, Tuple
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from github_rag.github_client import GitHubClient


class GitHubDataLoader:
    """Carregador de dados do GitHub para vectorização"""

    def __init__(self, github_client: GitHubClient):
        """
        Inicializa o carregador de dados do GitHub

        Args:
            github_client: Cliente GitHub autenticado para buscar dados
        """
        self.github_client = github_client
        self.issues_df = None
        self.code_files = None
        self.text_splitter = None

    def load_data(
        self,
        content_types: List[str],
        limit_issues: Optional[int] = None,
        max_files: Optional[int] = None,
    ) -> None:
        """
        Carrega dados do GitHub com base nos tipos de conteúdo especificados

        Args:
            content_types: Lista de tipos de conteúdo a carregar ('issue', 'code')
            limit_issues: Limite de issues a buscar
            max_files: Limite de arquivos de código a buscar
        """
        for content_type in content_types:
            self._load_content_type(content_type, limit_issues, max_files)

    def _load_content_type(
        self, content_type: str, limit_issues: Optional[int], max_files: Optional[int]
    ) -> None:
        """
        Carrega um tipo específico de conteúdo

        Args:
            content_type: Tipo de conteúdo ('issue', 'code')
            limit_issues: Limite de issues a buscar
            max_files: Limite de arquivos de código a buscar
        """
        if content_type == "issue":
            print("🔍 Buscando issues...")
            self.issues_df = pd.DataFrame(self.github_client.fetch_issues())
            print(
                f"✅ Encontrados {len(self.issues_df) if self.issues_df is not None else 0} issues"
            )

        elif content_type == "code":
            print("🔍 Buscando arquivos de código...")
            self.code_files = self.github_client.fetch_code_files()
            print(
                f"✅ Encontrados {len(self.code_files) if self.code_files else 0} arquivos de código"
            )

        else:
            print(f"⚠️ Tipo de conteúdo não suportado: {content_type}")

    def configure_text_splitter(
        self, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> None:
        """
        Configura o divisor de texto com os parâmetros especificados

        Args:
            chunk_size: Tamanho de cada chunk de texto
            chunk_overlap: Quantidade de sobreposição entre chunks
        """
        self.text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def create_text_chunks(
        self, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Cria chunks de texto a partir dos dados carregados

        Args:
            chunk_size: Tamanho de cada chunk de texto
            chunk_overlap: Quantidade de sobreposição entre chunks

        Returns:
            Lista de documentos processados com texto e metadados
        """
        # Configura o divisor de texto se ainda não estiver configurado
        if not self.text_splitter:
            self.configure_text_splitter(chunk_size, chunk_overlap)

        documents = []

        # Processa issues e código apenas se os dados existirem
        if self.issues_df is not None and not self.issues_df.empty:
            issue_documents = self._process_issues()
            documents.extend(issue_documents)

        if self.code_files:
            code_documents = self._process_code_files()
            documents.extend(code_documents)

        return documents

    def _process_issues(self) -> List[Dict[str, Any]]:
        """
        Processa issues e PRs com seus comentários

        Returns:
            Lista de documentos baseados em issues/PRs
        """
        if not self.text_splitter:
            self.configure_text_splitter()

        documents = []

        for _, row in self.issues_df.iterrows():
            # Determina se é issue ou PR e formata o texto adequadamente
            is_pr = "pull_request" in row
            item_type = "Pull Request" if is_pr else "Issue"
            source_type = "pull_request" if is_pr else "issue"
            item_number = row["number"]

            # Constrói o texto principal
            item_text = self._build_item_text(row, item_type, item_number)

            # Adiciona comentários, se existirem
            item_text = self._add_comments_to_text(item_text, row)

            # Divide o texto em chunks e adiciona aos documentos
            chunks = self.text_splitter.split_text(item_text)

            documents.extend(
                self._create_documents_from_chunks(
                    chunks,
                    source_type,
                    self._create_item_metadata(row, item_number, item_type),
                )
            )

        return documents

    def _build_item_text(self, row: pd.Series, item_type: str, item_number: int) -> str:
        """
        Constrói o texto principal de um issue ou PR

        Args:
            row: Linha do DataFrame com dados do item
            item_type: Tipo do item ('Issue' ou 'Pull Request')
            item_number: Número do item

        Returns:
            Texto formatado do item
        """
        item_text = (
            f"{item_type.upper()} #{item_number}: {row['title']}\n\n{row['body'] or ''}"
        )

        # Adiciona informações específicas de PR, se aplicável
        if item_type == "Pull Request":
            if "additions" in row and "deletions" in row:
                item_text += f"\nAdições: {row.get('additions', 0)}, Exclusões: {row.get('deletions', 0)}"
            if "merged" in row:
                item_text += f"\nStatus de Merge: {'Mesclado' if row.get('merged', False) else 'Não mesclado'}"

        return item_text

    def _add_comments_to_text(self, item_text: str, row: pd.Series) -> str:
        """
        Adiciona comentários ao texto do item

        Args:
            item_text: Texto atual do item
            row: Linha do DataFrame com dados do item

        Returns:
            Texto do item com comentários adicionados
        """
        comments = row.get("comments_data", [])

        if not comments or len(comments) == 0:
            return item_text

        item_text += "\n\n--- COMENTÁRIOS ---\n"

        for i, comment in enumerate(comments, 1):
            user = comment.get("user", "Usuário")
            body = comment.get("body", "")
            created_at = comment.get("created_at", "")

            item_text += f"\nCOMENTÁRIO #{i} por {user} em {created_at}:\n{body}\n"

        return item_text

    def _create_item_metadata(
        self, row: pd.Series, item_number: int, item_type: str
    ) -> Dict[str, Any]:
        """
        Cria metadados para um item (issue ou PR)

        Args:
            row: Linha do DataFrame com dados do item
            item_number: Número do item
            item_type: Tipo do item ('Issue' ou 'Pull Request')

        Returns:
            Dicionário com metadados
        """
        comments = row.get("comments_data", [])

        return {
            "source": "pull_request" if item_type == "Pull Request" else "issue",
            "number": item_number,
            "url": row["html_url"],
            "title": row["title"],
            "has_comments": len(comments) > 0,
            "type": item_type,
        }

    def _process_code_files(self) -> List[Dict[str, Any]]:
        """
        Processa arquivos de código

        Returns:
            Lista de documentos baseados em arquivos de código
        """
        if not self.text_splitter:
            self.configure_text_splitter()

        documents = []

        for file_info in self.code_files:
            content = file_info.get("content", "")

            if not content:
                continue

            # Adiciona informações do arquivo ao conteúdo para melhor contexto
            enhanced_content = self._enhance_code_content(content, file_info)

            # Divide o texto em chunks
            chunks = self.text_splitter.split_text(enhanced_content)

            # Cria metadados para o arquivo
            metadata = {
                "source": "code",
                "filename": file_info["name"],
                "url": file_info["url"],
                "extension": os.path.splitext(file_info["name"])[1],
            }

            documents.extend(
                self._create_documents_from_chunks(chunks, "code", metadata)
            )

        return documents

    def _enhance_code_content(self, content: str, file_info: Dict[str, Any]) -> str:
        """
        Adiciona informações contextuais ao conteúdo do código

        Args:
            content: Conteúdo do arquivo
            file_info: Informações sobre o arquivo

        Returns:
            Conteúdo aprimorado com informações de contexto
        """
        filename = file_info["name"]
        return f"ARQUIVO: {filename}\n\n{content}"

    def _create_documents_from_chunks(
        self, chunks: List[str], source_type: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Cria documentos a partir de chunks de texto

        Args:
            chunks: Lista de chunks de texto
            source_type: Tipo da fonte ('issue', 'pull_request', 'code')
            metadata: Metadados a serem incluídos em cada documento

        Returns:
            Lista de documentos formatados
        """
        documents = []

        for i, chunk in enumerate(chunks):
            # Copia os metadados para não modificar o original
            chunk_metadata = metadata.copy()

            # Adiciona informações sobre o chunk
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)

            documents.append(
                {
                    "text": chunk,
                    "metadata": chunk_metadata,
                }
            )

        return documents

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Retorna um resumo dos dados carregados

        Returns:
            Dicionário com resumo dos dados
        """
        summary = {
            "issues_count": len(self.issues_df) if self.issues_df is not None else 0,
            "code_files_count": len(self.code_files) if self.code_files else 0,
        }

        if self.issues_df is not None and not self.issues_df.empty:
            summary["issue_types"] = {
                "issues": sum(
                    1
                    for _, row in self.issues_df.iterrows()
                    if "pull_request" not in row
                ),
                "pull_requests": sum(
                    1 for _, row in self.issues_df.iterrows() if "pull_request" in row
                ),
            }

        if self.code_files:
            # Agrupar arquivos por extensão
            extensions = {}
            for file in self.code_files:
                ext = os.path.splitext(file["name"])[1]
                extensions[ext] = extensions.get(ext, 0) + 1

            summary["file_extensions"] = extensions

        return summary
