import requests
import pandas as pd
import os
import base64
from typing import List, Dict, Optional, Any


class GitHubClient:
    """Cliente para interagir com a API do GitHub"""

    def __init__(self, repo_url: str):
        self.repo_url = repo_url
        self.owner, self.repo = self._parse_repo_url(repo_url)
        self.api_base = f"https://api.github.com/repos/{self.owner}/{self.repo}"
        self.headers = {}
        self._setup_auth()

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
            self.headers["Authorization"] = f"Bearer {github_token}"

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
        issues = []
        page = 1

        print("🔍 Buscando issues...")

        while limit is None or len(issues) < limit:
            url = f"{self.api_base}/issues"
            params = {"state": state, "page": page, "per_page": 100}
            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code != 200:
                print(f"⚠️ Erro ao buscar issues: {response.status_code}")
                break

            page_issues = response.json()
            if not page_issues:
                break

            issues.extend(page_issues)
            page += 1
            print(f"📊 Encontrados {len(issues)} issues até agora...", end="\r")

        if limit is not None:
            issues = issues[:limit]

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
        comments = []
        page = 1

        while True:
            url = f"{self.api_base}/issues/{issue_number}/comments"
            params = {"page": page, "per_page": 100}

            try:
                response = requests.get(url, headers=self.headers, params=params)

                if response.status_code != 200:
                    print(
                        f"⚠️ Erro ao buscar comentários do issue #{issue_number}: {response.status_code}"
                    )
                    break

                page_comments = response.json()
                if not page_comments:
                    break

                comments.extend(page_comments)
                page += 1

            except Exception as e:
                print(
                    f"❌ Erro ao buscar comentários do issue #{issue_number}: {str(e)}"
                )
                break

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
            response = requests.get(url, headers=self.headers)

            if response.status_code != 200:
                print(f"⚠️ Falha ao acessar {path}: {response.status_code}")
                return []

            contents = response.json()
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
                    content = self._get_file_content(item["download_url"])
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

    def _get_file_content(self, download_url: str) -> str:
        """
        Baixa o conteúdo de um arquivo a partir da URL de download

        Args:
            download_url: URL para download do arquivo

        Returns:
            Conteúdo do arquivo como string ou string vazia em caso de falha
        """
        try:
            response = requests.get(download_url, headers=self.headers)
            if response.status_code == 200:
                return response.text
            print(
                f"⚠️ Falha ao baixar arquivo: {download_url} (Status: {response.status_code})"
            )
        except Exception as e:
            print(f"⚠️ Erro ao baixar arquivo: {download_url} - {str(e)}")

        return ""

    def _is_code_file(self, filename: str) -> bool:
        """Verifica se o arquivo é um arquivo de código"""
        extensions = [
            # Linguagens de uso geral
            ".py",
            ".pyc",
            ".pyd",
            ".pyo",
            ".pyw",
            ".pyz",  # Python
            ".js",
            ".mjs",
            ".cjs",  # JavaScript
            ".ts",
            ".tsx",  # TypeScript
            ".java",
            ".class",
            ".jar",  # Java
            ".c",
            ".h",  # C
            ".cpp",
            ".cc",
            ".cxx",
            ".hpp",
            ".hxx",
            ".h++",  # C++
            ".cs",  # C#
            ".php",
            ".phtml",
            ".php3",
            ".php4",
            ".php5",
            ".php7",
            ".phps",  # PHP
            ".rb",
            ".rbw",  # Ruby
            ".go",  # Go
            ".rs",
            ".rlib",  # Rust
            ".swift",  # Swift
            ".kt",
            ".kts",  # Kotlin
            ".scala",
            ".sc",  # Scala
            ".dart",  # Dart
            # Linguagens de script/shell
            ".sh",
            ".bash",
            ".zsh",
            ".fish",  # Shell scripts
            ".ps1",
            ".psm1",
            ".psd1",  # PowerShell
            ".bat",
            ".cmd",  # Batch (Windows)
            ".pl",
            ".pm",  # Perl
            ".lua",  # Lua
            ".r",
            ".rmd",  # R
            ".groovy",  # Groovy
            ".tcl",  # Tcl
            # Linguagens funcionais
            ".hs",
            ".lhs",  # Haskell
            ".erl",
            ".hrl",  # Erlang
            ".ex",
            ".exs",  # Elixir
            ".clj",
            ".cljs",
            ".cljc",  # Clojure
            ".lisp",
            ".cl",
            ".l",  # Common Lisp
            ".scm",
            ".ss",  # Scheme
            ".ml",
            ".mli",  # OCaml
            ".fs",
            ".fsi",
            ".fsx",  # F#
            # Linguagens específicas de domínio
            ".sql",  # SQL
            ".m",  # MATLAB/Objective-C
            ".f",
            ".f90",
            ".f95",
            ".f03",
            ".f08",  # Fortran
            ".d",  # D
            ".jl",  # Julia
            ".v",
            ".sv",  # Verilog/SystemVerilog
            ".vhd",
            ".vhdl",  # VHDL
            ".asm",
            ".s",  # Assembly
            ".cob",
            ".cbl",  # COBOL
            ".for",  # Fortran (antigo)
            ".pas",  # Pascal
            ".ada",
            ".adb",
            ".ads",  # Ada
            ".vb",  # Visual Basic
            # Linguagens de marcação/estilo
            ".html",
            ".htm",
            ".xhtml",  # HTML
            ".xml",
            ".xsl",
            ".xslt",  # XML
            ".css",
            ".scss",
            ".sass",
            ".less",  # CSS e pré-processadores
            ".json",
            ".jsonl",
            ".jsonc",  # JSON
            ".yaml",
            ".yml",  # YAML
            ".md",
            ".markdown",  # Markdown
            ".tex",
            ".sty",
            ".cls",  # LaTeX
            ".rst",  # reStructuredText
            ".toml",  # TOML
            ".haml",  # Haml
            ".jade",
            ".pug",  # Jade/Pug
            # Linguagens de template
            ".tmpl",
            ".template",  # Templates genéricos
            ".tpl",  # Templates Smarty
            ".j2",
            ".jinja",
            ".jinja2",  # Jinja
            ".vm",
            ".velocity",  # Velocity
            ".hbs",
            ".handlebars",
            ".mustache",  # Handlebars/Mustache
            ".erb",  # ERB (Ruby)
            ".jsp",  # JSP (Java)
            ".aspx",
            ".ascx",  # ASP.NET
            ".cshtml",
            ".razor",  # Razor (C#)
            # Arquivos de configuração/build
            ".ini",  # INI
            ".conf",
            ".config",  # Configurações
            ".make",
            ".mk",
            ".mak",  # Makefiles
            ".cmake",  # CMake
            ".gradle",  # Gradle
            ".sbt",  # SBT (Scala)
            ".ant",  # ANT
            ".prop",
            ".properties",  # Properties
            ".dockerfile",
            ".containerfile",  # Dockerfile
            ".dockerignore",  # Docker ignore
            ".tf",
            ".tfvars",  # Terraform
            ".proto",  # Protocol Buffers
            # Web/Mobile
            ".jsx",  # React JSX
            ".vue",  # Vue.js
            ".svelte",  # Svelte
            ".astro",  # Astro
            ".elm",  # Elm
            ".coffee",
            ".litcoffee",  # CoffeeScript
            ".as",  # ActionScript
            ".wsf",  # Windows Script File
            ".ejs",  # EJS
            ".wasm",  # WebAssembly
            ".wat",  # WebAssembly Text
            ".htaccess",  # Apache config
            ".xaml",  # XAML
            ".tsx",  # TypeScript React
            # Outros
            ".graphql",
            ".gql",  # GraphQL
            ".solidity",
            ".sol",  # Solidity
            ".ino",  # Arduino
            ".nix",  # Nix
            ".bf",  # Brainfuck
            ".nim",  # Nim
            ".re",
            ".rei",  # ReasonML
            ".zig",  # Zig
            ".cr",  # Crystal
            ".v",  # V
            ".pde",  # Processing
            ".inc",  # Include files
            ".ahk",  # AutoHotkey
            ".applescript",  # AppleScript
            ".purs",  # PureScript
            ".haxe",
            ".hx",  # Haxe
        ]

        return any(filename.endswith(ext) for ext in extensions)

    def get_file_content(self, file_info: Dict[str, str]) -> str:
        """Obtém o conteúdo de um arquivo"""
        response = requests.get(file_info["download_url"], headers=self.headers)
        if response.status_code == 200:
            return response.text
        return ""
