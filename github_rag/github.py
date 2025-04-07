from typing import Dict, List, Optional
import os
import requests
import base64
import multiprocessing
from urllib.parse import urlparse
import pickle
import hashlib


class GitHubDataExtractor:
    """Classe unificada para extração de dados estruturados do GitHub

    Atributos:
        api_token (str): Token de autenticação da API do GitHub
        repo_url (str): URL completa do repositório
        api_base (str): URL base da API GitHub
        cache_dir (str): Diretório para cache local
        session (requests.Session): Sessão HTTP reutilizável
    """

    def __init__(
        self, github_api_token: str, repo_url: str, cache_enabled: bool = True
    ):
        self.api_token = github_api_token
        self.repo_url = repo_url
        self.api_base = "https://api.github.com/repos"
        self.cache_enabled = cache_enabled
        self.cache_dir = "./.github_cache"
        self.session = self._create_session()

        # Setup do cache
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _create_session(self) -> requests.Session:
        """Cria sessão HTTP com headers de autenticação"""
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {self.api_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        return session

    def _make_request(self, url: str) -> Optional[Dict]:
        """Método genérico para requisições à API com cache e tratamento de erros"""
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        # Verificar cache
        if self.cache_enabled and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()

            # Salvar em cache
            if self.cache_enabled:
                with open(cache_path, "wb") as f:
                    pickle.dump(data, f)

            return data
        except Exception as e:
            print(f"Erro na requisição para {url}: {str(e)}")
            return None

    def _get_repo_info(self) -> Dict:
        """Extrai informações básicas do repositório"""
        path = f"{self.api_base}/{self._parse_repo_path()}"
        return self._make_request(path)

    def _parse_repo_path(self) -> str:
        """Converte URL do repositório para path da API"""
        parsed = urlparse(self.repo_url)
        return parsed.path.strip("/")

    def _parallel_fetch(self, urls: List[str]) -> List[Dict]:
        """Executa fetch paralelo de múltiplos endpoints"""
        with multiprocessing.Pool() as pool:
            results = pool.map(self._make_request, urls)
        return [r for r in results if r is not None]

    def get_structured_data(self, content_types: List[str]) -> Dict[str, List[Dict]]:
        """Método principal para coleta estruturada de dados"""
        repo_info = self._get_repo_info()
        if not repo_info:
            raise ValueError("Repositório não encontrado ou sem acesso")

        results = {}

        if "issues" in content_types:
            results["issues"] = self._get_issues_with_comments()

        if "pull_requests" in content_types:
            results["pull_requests"] = self._get_pull_requests_with_comments()

        if "releases" in content_types:
            results["releases"] = self._get_releases()

        if "documentation" in content_types:
            results["documentation"] = self._get_documentation_files()

        return results

    def _get_issues_with_comments(self) -> List[Dict]:
        """Extrai issues e seus comentários formatados"""
        issues_url = f"{self.api_base}/{self._parse_repo_path()}/issues?state=all"
        issues = self._make_request(issues_url) or []

        # Paralelizar busca de comentários
        comment_urls = [issue["comments_url"] for issue in issues]
        comments = self._parallel_fetch(comment_urls)

        structured_issues = []
        for issue, comment_list in zip(issues, comments):
            structured_issues.append(
                {
                    "id": issue["id"],
                    "type": "issue",
                    "title": issue["title"],
                    "body": issue["body"],
                    "state": issue["state"],
                    "created_at": issue["created_at"],
                    "updated_at": issue["updated_at"],
                    "labels": [label["name"] for label in issue.get("labels", [])],
                    "comments": [
                        {
                            "author": comment["user"]["login"],
                            "body": comment["body"],
                            "created_at": comment["created_at"],
                        }
                        for comment in comment_list
                    ],
                    "metadata": {
                        "api_url": issue["url"],
                        "html_url": issue["html_url"],
                    },
                }
            )

        return structured_issues

    def _get_pull_requests_with_comments(self) -> List[Dict]:
        """Extrai PRs com comentários e revisões"""
        prs_url = f"{self.api_base}/{self._parse_repo_path()}/pulls?state=all"
        prs = self._make_request(prs_url) or []

        structured_prs = []
        for pr in prs:
            # Buscar dados complementares
            comments = self._make_request(pr["comments_url"]) or []
            reviews = self._make_request(pr["review_comments_url"]) or []

            structured_prs.append(
                {
                    "id": pr["id"],
                    "type": "pull_request",
                    "title": pr["title"],
                    "body": pr["body"],
                    "state": pr["state"],
                    "created_at": pr["created_at"],
                    "updated_at": pr["updated_at"],
                    "merge_commit_sha": pr.get("merge_commit_sha"),
                    "comments": [
                        {
                            "type": "comment",
                            "author": c["user"]["login"],
                            "body": c["body"],
                            "created_at": c["created_at"],
                        }
                        for c in comments
                    ],
                    "reviews": [
                        {
                            "review_id": r["id"],
                            "type": "review",
                            "author": r["user"]["login"],
                            "author_association": r["author_association"],
                            "body": r["body"],
                            "timestamps": {
                                "created": r["created_at"],
                                "updated": r["updated_at"],
                            },
                            "metadata": {
                                "commit_id": r["commit_id"],
                                "html_url": r["html_url"],
                                "pull_request_url": r["pull_request_url"],
                                "node_id": r["node_id"],
                            },
                            "interactions": {"reactions": r.get("reactions", {})},
                        }
                        for r in reviews
                        if r and isinstance(r, dict)
                    ],
                    "metadata": {
                        "base_branch": pr["base"]["ref"],
                        "head_branch": pr["head"]["ref"],
                        "api_url": pr["url"],
                        "html_url": pr["html_url"],
                    },
                }
            )

        return structured_prs

    def _get_releases(self) -> List[Dict]:
        """Extrai informações de releases"""
        releases_url = f"{self.api_base}/{self._parse_repo_path()}/releases"
        releases = self._make_request(releases_url) or []

        return [
            {
                "id": release["id"],
                "type": "release",
                "tag_name": release["tag_name"],
                "name": release["name"],
                "body": release["body"],
                "created_at": release["created_at"],
                "published_at": release["published_at"],
                "assets": [
                    {
                        "name": asset["name"],
                        "size": asset["size"],
                        "download_url": asset["browser_download_url"],
                    }
                    for asset in release.get("assets", [])
                ],
                "metadata": {
                    "api_url": release["url"],
                    "html_url": release["html_url"],
                },
            }
            for release in releases
        ]

    def _get_documentation_files(self) -> List[Dict]:
        """Baixa e processa arquivos de documentação"""
        repo_content_url = f"{self.api_base}/{self._parse_repo_path()}/contents/"
        all_files = self._make_request(repo_content_url) or []

        doc_extensions = {".md", ".mdx", ".html", ".tex", ".pdf"}
        doc_files = [
            file
            for file in all_files
            if os.path.splitext(file["name"])[1].lower() in doc_extensions
        ]

        # Processamento paralelo dos conteúdos
        with multiprocessing.Pool() as pool:
            contents = pool.map(self._process_doc_file, doc_files)

        return [c for c in contents if c is not None]

    def _process_doc_file(self, file_info: Dict) -> Optional[Dict]:
        """Processa individualmente cada arquivo de documentação"""
        content_url = file_info["url"]
        content_data = self._make_request(content_url)

        if not content_data or "content" not in content_data:
            return None

        content = base64.b64decode(content_data["content"]).decode(
            "utf-8", errors="replace"
        )

        return {
            "type": "documentation",
            "path": file_info["path"],
            "name": file_info["name"],
            "content": content,
            "encoding": "utf-8",
            "size": file_info["size"],
            "metadata": {
                "sha": file_info["sha"],
                "html_url": file_info["html_url"],
                "download_url": file_info["download_url"],
            },
        }
