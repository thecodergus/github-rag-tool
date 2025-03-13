import requests
import pandas as pd
import os
import base64
from typing import List, Dict, Optional, Any
import time


class GitHubClient:
    """Cliente para interagir com a API do GitHub"""

    def __init__(self, repo_url: str):
        self.repo_url = repo_url
        self.owner, self.repo = self._parse_repo_url(repo_url)
        self.api_base = f"https://api.github.com/repos/{self.owner}/{self.repo}"
        self.headers = {"User-Agent": "request"}
        self._setup_auth()

        # Definir categorias de extensões de arquivo
        self._code_extensions = self._initialize_code_extensions()

    def _parse_repo_url(self, url: str) -> tuple:
        """Extrai owner e repo da URL do GitHub"""
        parts = url.strip("/").split("/")
        if "github.com" in parts:
            idx = parts.index("github.com")
            return parts[idx + 1], parts[idx + 2]
        return parts[-2], parts[-1]

    def _setup_auth(self):
        """Configura autenticação com GitHub API"""
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            self.headers["Authorization"] = f"token {github_token}"

    def _initialize_code_extensions(self) -> Dict[str, List[str]]:
        """Inicializa as extensões de arquivos de código organizadas por categoria"""
        return {
            "general": [
                ".py",
                ".pyc",
                ".pyd",
                ".pyo",
                ".pyw",
                ".pyz",
                ".js",
                ".mjs",
                ".cjs",
                ".ts",
                ".tsx",
                ".java",
                ".class",
                ".jar",
                ".c",
                ".h",
                ".cpp",
                ".cc",
                ".cxx",
                ".hpp",
                ".hxx",
                ".h++",
                ".cs",
                ".php",
                ".phtml",
                ".php3",
                ".php4",
                ".php5",
                ".php7",
                ".phps",
                ".rb",
                ".rbw",
                ".go",
                ".rs",
                ".rlib",
                ".swift",
                ".kt",
                ".kts",
                ".scala",
                ".sc",
                ".dart",
            ],
            "script": [
                ".sh",
                ".bash",
                ".zsh",
                ".fish",
                ".ps1",
                ".psm1",
                ".psd1",
                ".bat",
                ".cmd",
                ".pl",
                ".pm",
                ".lua",
                ".r",
                ".rmd",
                ".groovy",
                ".tcl",
            ],
            "functional": [
                ".hs",
                ".lhs",
                ".erl",
                ".hrl",
                ".ex",
                ".exs",
                ".clj",
                ".cljs",
                ".cljc",
                ".lisp",
                ".cl",
                ".l",
                ".scm",
                ".ss",
                ".ml",
                ".mli",
                ".fs",
                ".fsi",
                ".fsx",
            ],
            "domain_specific": [
                ".sql",
                ".m",
                ".f",
                ".f90",
                ".f95",
                ".f03",
                ".f08",
                ".d",
                ".jl",
                ".v",
                ".sv",
                ".vhd",
                ".vhdl",
                ".asm",
                ".s",
                ".cob",
                ".cbl",
                ".for",
                ".pas",
                ".ada",
                ".adb",
                ".ads",
                ".vb",
            ],
            "markup": [
                ".html",
                ".htm",
                ".xhtml",
                ".xml",
                ".xsl",
                ".xslt",
                ".css",
                ".scss",
                ".sass",
                ".less",
                ".json",
                ".jsonl",
                ".jsonc",
                ".yaml",
                ".yml",
                ".md",
                ".markdown",
                ".tex",
                ".sty",
                ".cls",
                ".rst",
                ".toml",
                ".haml",
                ".jade",
                ".pug",
            ],
            "template": [
                ".tmpl",
                ".template",
                ".tpl",
                ".j2",
                ".jinja",
                ".jinja2",
                ".vm",
                ".velocity",
                ".hbs",
                ".handlebars",
                ".mustache",
                ".erb",
                ".jsp",
                ".aspx",
                ".ascx",
                ".cshtml",
                ".razor",
            ],
            "config": [
                ".ini",
                ".conf",
                ".config",
                ".make",
                ".mk",
                ".mak",
                ".cmake",
                ".gradle",
                ".sbt",
                ".ant",
                ".prop",
                ".properties",
                ".dockerfile",
                ".containerfile",
                ".dockerignore",
                ".tf",
                ".tfvars",
                ".proto",
            ],
            "web": [
                ".jsx",
                ".vue",
                ".svelte",
                ".astro",
                ".elm",
                ".coffee",
                ".litcoffee",
                ".as",
                ".wsf",
                ".ejs",
                ".wasm",
                ".wat",
                ".htaccess",
                ".xaml",
                ".tsx",
            ],
            "other": [
                ".graphql",
                ".gql",
                ".solidity",
                ".sol",
                ".ino",
                ".nix",
                ".bf",
                ".nim",
                ".re",
                ".rei",
                ".zig",
                ".cr",
                ".v",
                ".pde",
                ".inc",
                ".ahk",
                ".applescript",
                ".purs",
                ".haxe",
                ".hx",
            ],
        }

    def _make_request(
        self, url: str, params: Dict = None, error_message: str = "Erro na requisição"
    ) -> Optional[Dict]:
        """
        Faz uma requisição à API do GitHub com tratamento padronizado de erros

        Args:
            url: URL para a requisição
            params: Parâmetros da requisição
            error_message: Mensagem de erro personalizada

        Returns:
            Resposta da API em formato JSON ou None em caso de erro
        """
        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code != 200:
                print(f"⚠️ {error_message}: {response.status_code}")
                return None
            return response.json()
        except Exception as e:
            print(f"❌ {error_message}: {str(e)}")
            return None

    def _paginated_request(
        self,
        url: str,
        params: Dict = None,
        limit: Optional[int] = None,
        item_name: str = "items",
        sleep_time: float = 0.5,
    ) -> List[Dict]:
        """
        Faz requisições paginadas à API do GitHub

        Args:
            url: URL base para a requisição
            params: Parâmetros adicionais da requisição
            limit: Número máximo de itens a serem buscados
            item_name: Nome dos itens para mensagens de log
            sleep_time: Tempo de espera entre requisições

        Returns:
            Lista de itens das respostas
        """
        items = []
        page = 1
        params = params or {}

        while limit is None or len(items) < limit:
            params["page"] = page
            params["per_page"] = 100

            data = self._make_request(
                url, params=params, error_message=f"Erro ao buscar {item_name}"
            )

            if not data:
                break

            if not data:  # Lista vazia
                break

            items.extend(data)
            page += 1

            print(f"📊 Encontrados {len(items)} {item_name} até agora...", end="\r")
            time.sleep(sleep_time)

        if limit is not None:
            items = items[:limit]

        return items

    def get_file_content(self, file_info: Dict[str, str]) -> str:
        """
        Obtém o conteúdo de um arquivo

        Args:
            file_info: Dicionário com informações do arquivo, incluindo 'download_url'

        Returns:
            Conteúdo do arquivo como string
        """
        try:
            download_url = file_info["download_url"]
            response = requests.get(download_url, headers=self.headers)
            if response.status_code == 200:
                return response.text
            print(
                f"⚠️ Falha ao baixar arquivo: {download_url} (Status: {response.status_code})"
            )
        except Exception as e:
            print(f"⚠️ Erro ao baixar arquivo: {str(e)}")
        return ""

    def _is_code_file(self, filename: str) -> bool:
        """
        Verifica se o arquivo é um arquivo de código

        Args:
            filename: Nome do arquivo

        Returns:
            True se for um arquivo de código, False caso contrário
        """
        # Combina todas as extensões em uma única lista
        all_extensions = []
        for category_extensions in self._code_extensions.values():
            all_extensions.extend(category_extensions)

        return any(filename.endswith(ext) for ext in all_extensions)

    def fetch_issues(
        self, state: str = "all", limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Busca issues do repositório e seus comentários

        Args:
            state: Estado dos issues ('open', 'closed', 'all')
            limit: Número máximo de issues a serem buscados (None para todos)

        Returns:
            DataFrame com issues e seus comentários
        """
        print("🔍 Buscando issues...")

        url = f"{self.api_base}/issues"
        params = {"state": state}

        issues = self._paginated_request(
            url, params=params, limit=limit, item_name="issues"
        )

        print(f"\n✅ Total de {len(issues)} issues encontrados")

        # Converter para DataFrame e processar
        if not issues:
            return pd.DataFrame()

        df = pd.DataFrame(issues)

        # Selecionar colunas relevantes
        if set(["number", "title", "body", "state", "created_at", "html_url"]).issubset(
            df.columns
        ):
            df = df[["number", "title", "body", "state", "created_at", "html_url"]]
            df["created_at"] = pd.to_datetime(df["created_at"])

            # Buscar comentários para cada issue
            print("🔍 Buscando comentários para cada issue...")
            total = len(df)

            # Criar coluna de comentários
            df["comments_data"] = None

            for idx, row in df.iterrows():
                issue_number = row["number"]
                print(
                    f"💬 Buscando comentários do issue #{issue_number} ({idx+1}/{total})",
                    end="\r",
                )

                comments = self.fetch_issue_comments(issue_number)
                df.at[idx, "comments_data"] = comments

            print(f"\n✅ Comentários buscados para {total} issues")

        return df

    def fetch_issue_comments(self, issue_number: int) -> List[Dict[str, Any]]:
        """
        Busca todos os comentários de um issue específico

        Args:
            issue_number: Número do issue

        Returns:
            Lista de comentários com seus metadados
        """
        url = f"{self.api_base}/issues/{issue_number}/comments"

        comments = self._paginated_request(
            url, item_name=f"comentários do issue #{issue_number}", sleep_time=0.5
        )

        # Processar e formatar comentários
        formatted_comments = []
        for comment in comments:
            formatted_comments.append(
                {
                    "id": comment["id"],
                    "user": comment["user"]["login"],
                    "body": comment["body"],
                    "created_at": comment["created_at"],
                    "html_url": comment["html_url"],
                }
            )

        return formatted_comments

    def fetch_code_files(
        self, path: str = "", max_files: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca arquivos de código no repositório recursivamente.

        Args:
            path: Caminho dentro do repositório para buscar (vazio = raiz)
            max_files: Número máximo de arquivos a serem buscados (None para sem limite)

        Returns:
            Lista de dicionários contendo informações e conteúdo dos arquivos
        """
        url = f"{self.api_base}/contents/{path}"

        try:
            print(f"🔍 Explorando diretório: {path or 'raiz'}")

            contents = self._make_request(url, error_message=f"Falha ao acessar {path}")

            if not contents:
                return []

            files = []
            dirs = []

            total_items = len(contents)
            processed = 0

            for item in contents:
                processed += 1
                if path:  # Só mostra progresso em subdiretórios
                    print(
                        f"📂 Processando em {path}: {processed}/{total_items}", end="\r"
                    )

                if item["type"] == "file" and self._is_code_file(item["name"]):
                    print(f"📄 Baixando: {item['path']}")
                    content = self.get_file_content(item)
                    files.append(
                        {
                            "name": item["path"],
                            "path": item["path"],
                            "download_url": item["download_url"],
                            "url": item["html_url"],
                            "sha": item["sha"],
                            "content": content,
                        }
                    )
                elif item["type"] == "dir":
                    dirs.append(item["path"])

                time.sleep(0.5)

            if path:
                print()  # Nova linha após terminar o processamento do diretório

            # Recursivamente buscar em subdiretórios sem limitação
            for dir_path in dirs:
                subdir_files = self.fetch_code_files(dir_path, max_files=None)
                files.extend(subdir_files)

            return files

        except Exception as e:
            print(f"❌ Erro ao processar diretório {path}: {str(e)}")
            return []
