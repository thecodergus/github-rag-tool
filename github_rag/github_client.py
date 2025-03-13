import os
import time
import json
import hashlib
import requests
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import logging

from github_rag.utils import parse_github_repo_url


class GitHubClient:
    """
    Cliente para interação com a API do GitHub com tratamento avançado de limites de taxa,
    cache de requisições e backoff exponencial.

    Otimizado para repositórios de robótica como o LeRobot (https://github.com/huggingface/lerobot),
    fornecendo métodos específicos para análise de issues, PRs e código-fonte.
    """

    def __init__(
        self,
        repo_url: str,
        token: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: str = ".github_cache",
        cache_ttl: int = 86400,  # 24 horas em segundos
        log_level: int = logging.INFO,
    ):
        self.repo_url = repo_url
        self.owner, self.repo = parse_github_repo_url(repo_url)
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "requests",
        }

        # Configuração de autenticação
        if token:
            self.headers["Authorization"] = f"token {token}"
        else:
            print(
                "⚠️ Operando sem token de autenticação. Limites de taxa serão mais restritivos."
            )

        # Configuração de cache
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl

        if use_cache and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Configuração de logging
        self.logger = self._setup_logger(log_level)

        # Estatísticas de uso da API
        self.requests_made = 0
        self.cache_hits = 0
        self.rate_limit_hits = 0
        self.last_response = None

        # Informações do repositório
        self.repo_info = self._fetch_repo_info()

    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Configura o logger para a classe."""
        logger = logging.getLogger(f"GitHubClient-{self.owner}-{self.repo}")
        logger.setLevel(log_level)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _get_cache_key(self, url: str, params: Dict = None) -> str:
        """Gera uma chave de cache única para uma requisição."""
        params_str = json.dumps(params or {}, sort_keys=True)
        key = f"{url}_{params_str}"
        return hashlib.md5(key.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Recupera dados do cache se disponíveis e válidos."""
        if not self.use_cache:
            return None

        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file):
            # Verificar idade do cache
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < self.cache_ttl:
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        self.cache_hits += 1
                        self.logger.debug(f"Cache hit para {cache_key}")
                        return json.load(f)
                except Exception as e:
                    self.logger.warning(f"Erro ao ler cache: {str(e)}")
            else:
                self.logger.debug(
                    f"Cache expirado para {cache_key} (idade: {cache_age:.1f}s)"
                )

        return None

    def _save_to_cache(self, cache_key: str, data: Dict) -> None:
        """Salva dados no cache."""
        if not self.use_cache:
            return

        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
                self.logger.debug(f"Dados salvos em cache: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Erro ao salvar cache: {str(e)}")

    def check_rate_limit(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Verifica os limites de taxa atuais da API.

        Returns:
            Tuple[int, int]: (requisições restantes, timestamp para reset)
        """
        url = f"{self.base_url}/rate_limit"

        try:
            response = requests.get(url, headers=self.headers)
            self.last_response = response

            if response.status_code == 200:
                data = response.json()

                # Obter limites de taxa core e search
                core_rate = data["resources"]["core"]
                search_rate = data["resources"]["search"]

                # Formatar horário de reset para exibição
                core_reset_time = datetime.fromtimestamp(core_rate["reset"]).strftime(
                    "%H:%M:%S"
                )
                search_reset_time = datetime.fromtimestamp(
                    search_rate["reset"]
                ).strftime("%H:%M:%S")

                # Informar limites de taxa
                self.logger.info("\n--- Limites de Taxa da API GitHub ---")
                self.logger.info(
                    f"📊 Core API: {core_rate['remaining']}/{core_rate['limit']} restantes"
                )
                self.logger.info(
                    f"🔎 Search API: {search_rate['remaining']}/{search_rate['limit']} restantes"
                )
                self.logger.info(f"⏱️ Core API reset às: {core_reset_time}")
                self.logger.info(f"⏱️ Search API reset às: {search_reset_time}")

                # Avisar se estiver próximo do limite
                if core_rate["remaining"] < (core_rate["limit"] * 0.1):
                    self.logger.warning(
                        f"⚠️ ATENÇÃO: Menos de 10% das requisições Core disponíveis!"
                    )

                return core_rate["remaining"], core_rate["reset"]
            else:
                self.logger.error(
                    f"❌ Erro ao verificar limites de taxa: {response.status_code}"
                )
                return None, None

        except Exception as e:
            self.logger.error(f"❌ Exceção ao verificar limites de taxa: {str(e)}")
            return None, None

    def _make_request(
        self,
        url: str,
        params: Optional[Dict] = None,
        method: str = "GET",
        data: Optional[Dict] = None,
        error_message: str = "Erro na requisição",
        max_retries: int = 5,
        use_cache: Optional[bool] = None,
    ) -> Optional[Dict]:
        """
        Realiza uma requisição HTTP com tratamento avançado de erros e limites de taxa.

        Args:
            url: URL completa da requisição
            params: Parâmetros de query string
            method: Método HTTP (GET, POST, etc)
            data: Dados para enviar (para POST, PUT, etc)
            error_message: Mensagem personalizada para erros
            max_retries: Número máximo de tentativas
            use_cache: Sobrescreve configuração global de cache

        Returns:
            Dict: Resposta da API em formato JSON ou None em caso de erro
        """
        use_cache = self.use_cache if use_cache is None else use_cache

        # Gerar chave de cache e verificar se temos dados em cache
        if method == "GET" and use_cache:
            cache_key = self._get_cache_key(url, params)
            cached_data = self._get_from_cache(cache_key)

            if cached_data:
                self.logger.debug(f"🔄 Usando dados em cache para: {url}")
                return cached_data

        # Fazer a requisição com retentativas
        retries = 0
        self.requests_made += 1

        while retries < max_retries:
            try:
                if method == "GET":
                    response = requests.get(
                        url, headers=self.headers, params=params, timeout=30
                    )
                elif method == "POST":
                    response = requests.post(
                        url, headers=self.headers, params=params, json=data, timeout=30
                    )
                elif method == "PUT":
                    response = requests.put(
                        url, headers=self.headers, params=params, json=data, timeout=30
                    )
                elif method == "DELETE":
                    response = requests.delete(
                        url, headers=self.headers, params=params, timeout=30
                    )
                else:
                    raise ValueError(f"Método HTTP não suportado: {method}")

                # Salvar a última resposta para uso em outros métodos
                self.last_response = response

                # Extrair informações de limite de taxa dos cabeçalhos
                remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
                limit = int(response.headers.get("X-RateLimit-Limit", 0))
                reset_time = int(response.headers.get("X-RateLimit-Reset", 0))

                # Se for bem-sucedido, retornar dados e cachear
                if response.status_code == 200 or response.status_code == 201:
                    result = response.json()

                    # Salvar em cache se for GET
                    if method == "GET" and use_cache:
                        self._save_to_cache(cache_key, result)

                    # Avisar se estiver com poucas requisições restantes
                    if remaining < (limit * 0.1) and limit > 0:
                        reset_datetime = datetime.fromtimestamp(reset_time).strftime(
                            "%H:%M:%S"
                        )
                        self.logger.warning(
                            f"⚠️ Apenas {remaining}/{limit} requisições restantes até {reset_datetime}"
                        )

                    return result

                # Tratar limites de taxa (403/429)
                elif response.status_code in (403, 429):
                    self.rate_limit_hits += 1

                    # Verificar se é realmente um problema de limite de taxa
                    if (
                        "X-RateLimit-Remaining" in response.headers
                        and int(response.headers["X-RateLimit-Remaining"]) == 0
                    ):
                        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                        current_time = time.time()
                        sleep_time = (
                            max(reset_time - current_time, 0) + 2
                        )  # Margem de segurança

                        self.logger.warning(
                            f"⏳ Limite de taxa atingido. Aguardando {sleep_time:.1f} segundos até reset..."
                        )
                        time.sleep(sleep_time)
                        retries += 1
                        continue

                    # Se for 429, usar o header Retry-After se disponível
                    if (
                        response.status_code == 429
                        and "Retry-After" in response.headers
                    ):
                        retry_after = int(response.headers["Retry-After"])
                        self.logger.warning(
                            f"⏳ Taxa excedida. Aguardando {retry_after} segundos conforme solicitado."
                        )
                        time.sleep(retry_after)
                        retries += 1
                        continue

                    # Backoff exponencial para outras tentativas
                    wait_time = (2**retries) + (
                        time.time() % 1
                    )  # Adiciona um pouco de aleatoriedade
                    self.logger.warning(
                        f"⏳ {error_message} ({response.status_code}). Tentativa {retries+1}/{max_retries} em {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                    retries += 1

                # Outros erros
                else:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get("message", "Sem detalhes do erro")
                    self.logger.error(
                        f"❌ {error_message}: {response.status_code} - {error_msg}"
                    )

                    # Alguns erros não devem ser retentados
                    if response.status_code in (401, 404, 422):
                        return None

                    # Para outros erros, tentar novamente com backoff
                    wait_time = (2**retries) + (time.time() % 1)
                    self.logger.warning(
                        f"⏳ Tentativa {retries+1}/{max_retries} em {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                    retries += 1

            except requests.exceptions.RequestException as e:
                self.logger.error(f"❌ Erro de conexão: {str(e)}")
                wait_time = (2**retries) + (time.time() % 1)
                self.logger.warning(
                    f"⏳ Tentativa {retries+1}/{max_retries} em {wait_time:.1f}s"
                )
                time.sleep(wait_time)
                retries += 1

        # Se chegou aqui, todas as tentativas falharam
        self.logger.error(f"❌ Falha após {max_retries} tentativas para: {url}")
        return None

    def _fetch_repo_info(self) -> Optional[Dict]:
        """
        Busca informações básicas sobre o repositório.

        Returns:
            Dict: Informações do repositório ou None em caso de erro
        """
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}"
        return self._make_request(
            url, error_message="Erro ao obter informações do repositório"
        )

    def fetch_issues(
        self,
        state: str = "all",
        per_page: int = 100,
        since: Optional[str] = None,
        labels: Optional[str] = None,
        sort: str = "created",
        direction: str = "desc",
    ) -> List[Dict]:
        """
        Obtém todas as issues de um repositório, lidando com paginação.

        Args:
            state: Estado das issues (open, closed, all)
            per_page: Número de itens por página
            since: Data ISO 8601 (YYYY-MM-DDTHH:MM:SSZ) para filtrar issues atualizadas após esta data
            labels: Lista de labels separadas por vírgula
            sort: Campo para ordenação (created, updated, comments)
            direction: Direção da ordenação (asc, desc)

        Returns:
            List[Dict]: Lista com todas as issues
        """
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/issues"
        params = {
            "state": state,
            "per_page": per_page,
            "page": 1,
            "sort": sort,
            "direction": direction,
        }

        if since:
            params["since"] = since

        if labels:
            params["labels"] = labels

        all_issues = []
        total_pages = 0

        self.logger.info(
            f"🔍 Buscando issues ({state}) do repositório {self.owner}/{self.repo}"
        )

        while True:
            issues = self._make_request(
                url,
                params=params,
                error_message=f"Erro ao obter issues da página {params['page']}",
            )

            if not issues or len(issues) == 0:
                break

            # Filtra para remover PRs (a API do GitHub retorna PRs como issues)
            filtered_issues = [issue for issue in issues if "pull_request" not in issue]
            all_issues.extend(filtered_issues)

            total_pages = params["page"]

            self.logger.info(
                f"📋 Página {params['page']}: {len(filtered_issues)} issues encontradas"
            )

            # Verificar se tem mais páginas
            if len(issues) < per_page:
                break

            params["page"] += 1

            # Pequena pausa entre requisições para evitar sobrecarga
            time.sleep(0.25)

        self.logger.info(
            f"✅ Total de issues coletadas: {len(all_issues)} em {total_pages} páginas"
        )
        return all_issues

    def fetch_pull_requests(
        self,
        state: str = "all",
        per_page: int = 100,
        sort: str = "created",
        direction: str = "desc",
        base: Optional[str] = None,
    ) -> List[Dict]:
        """
        Obtém todos os pull requests de um repositório, lidando com paginação.

        Args:
            state: Estado dos PRs (open, closed, all)
            per_page: Número de itens por página
            sort: Campo para ordenação (created, updated, popularity, long-running)
            direction: Direção da ordenação (asc, desc)
            base: Filtrar PRs por branch base

        Returns:
            List[Dict]: Lista com todos os pull requests
        """
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/pulls"
        params = {
            "state": state,
            "per_page": per_page,
            "page": 1,
            "sort": sort,
            "direction": direction,
        }

        if base:
            params["base"] = base

        all_prs = []
        total_pages = 0

        self.logger.info(
            f"🔍 Buscando pull requests ({state}) do repositório {self.owner}/{self.repo}"
        )

        while True:
            prs = self._make_request(
                url,
                params=params,
                error_message=f"Erro ao obter PRs da página {params['page']}",
            )

            if not prs or len(prs) == 0:
                break

            all_prs.extend(prs)
            total_pages = params["page"]

            self.logger.info(f"📋 Página {params['page']}: {len(prs)} PRs encontrados")

            # Verificar se tem mais páginas
            if len(prs) < per_page:
                break

            params["page"] += 1

            # Pequena pausa entre requisições para evitar sobrecarga
            time.sleep(0.25)

        self.logger.info(
            f"✅ Total de pull requests coletados: {len(all_prs)} em {total_pages} páginas"
        )
        return all_prs

    def fetch_pr_details(self, pr_number: int) -> Optional[Dict]:
        """
        Obtém detalhes de um pull request específico, incluindo reviews e commits.

        Args:
            pr_number: Número do pull request

        Returns:
            Dict: Dados detalhados do pull request ou None em caso de erro
        """
        # Buscar dados básicos do PR
        pr_url = f"{self.base_url}/repos/{self.owner}/{self.repo}/pulls/{pr_number}"
        pr_data = self._make_request(
            pr_url, error_message=f"Erro ao obter PR #{pr_number}"
        )

        if not pr_data:
            return None

        # Buscar commits do PR
        commits_url = f"{pr_url}/commits"
        commits = self._make_request(
            commits_url, error_message=f"Erro ao obter commits do PR #{pr_number}"
        )

        # Buscar reviews do PR
        reviews_url = f"{pr_url}/reviews"
        reviews = self._make_request(
            reviews_url, error_message=f"Erro ao obter reviews do PR #{pr_number}"
        )

        # Agregar dados
        pr_data["commits_data"] = commits or []
        pr_data["reviews_data"] = reviews or []

        return pr_data

    def fetch_commits(
        self,
        per_page: int = 100,
        since: Optional[str] = None,
        until: Optional[str] = None,
        path: Optional[str] = None,
        author: Optional[str] = None,
    ) -> List[Dict]:
        """
        Obtém todos os commits de um repositório, lidando com paginação.

        Args:
            per_page: Número de itens por página
            since: Data ISO 8601 para filtrar commits a partir desta data (YYYY-MM-DDTHH:MM:SSZ)
            until: Data ISO 8601 para filtrar commits até esta data (YYYY-MM-DDTHH:MM:SSZ)
            path: Filtra commits que modificaram arquivos neste caminho
            author: Filtra commits por autor (username GitHub)

        Returns:
            List[Dict]: Lista com todos os commits
        """
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/commits"
        params = {"per_page": per_page, "page": 1}

        if since:
            params["since"] = since

        if until:
            params["until"] = until

        if path:
            params["path"] = path

        if author:
            params["author"] = author

        all_commits = []
        total_pages = 0

        self.logger.info(f"🔍 Buscando commits do repositório {self.owner}/{self.repo}")

        while True:
            commits = self._make_request(
                url,
                params=params,
                error_message=f"Erro ao obter commits da página {params['page']}",
            )

            if not commits or len(commits) == 0:
                break

            all_commits.extend(commits)
            total_pages = params["page"]

            self.logger.info(
                f"📋 Página {params['page']}: {len(commits)} commits encontrados"
            )

            # Verificar se tem mais páginas
            if len(commits) < per_page:
                break

            params["page"] += 1

            # Pequena pausa entre requisições para evitar sobrecarga
            time.sleep(0.25)

        self.logger.info(
            f"✅ Total de commits coletados: {len(all_commits)} em {total_pages} páginas"
        )
        return all_commits

    def fetch_commit_details(self, commit_sha: str) -> Optional[Dict]:
        """
        Obtém detalhes de um commit específico, incluindo alterações.

        Args:
            commit_sha: SHA do commit

        Returns:
            Dict: Dados detalhados do commit ou None em caso de erro
        """
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/commits/{commit_sha}"
        return self._make_request(
            url, error_message=f"Erro ao obter detalhes do commit {commit_sha}"
        )

    def search_repositories(
        self, query: str, sort: str = "stars", order: str = "desc", per_page: int = 100
    ) -> List[Dict]:
        """
        Pesquisa repositórios no GitHub.

        Args:
            query: Query de pesquisa
            sort: Campo para ordenação (stars, forks, updated)
            order: Direção da ordenação (asc, desc)
            per_page: Itens por página

        Returns:
            List[Dict]: Lista de repositórios encontrados
        """
        url = f"{self.base_url}/search/repositories"
        params = {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": per_page,
            "page": 1,
        }

        all_repos = []
        total_count = 0

        while True:
            result = self._make_request(
                url,
                params=params,
                error_message=f"Erro na pesquisa de repositórios página {params['page']}",
            )

            if not result or "items" not in result:
                break

            if params["page"] == 1:
                total_count = result.get("total_count", 0)
                self.logger.info(f"📊 Total de resultados encontrados: {total_count}")

            items = result["items"]
            all_repos.extend(items)

            # Verificar se tem mais páginas
            if len(items) < per_page or len(all_repos) >= total_count:
                break

            params["page"] += 1

            # Pequena pausa entre requisições para evitar sobrecarga
            time.sleep(0.5)  # Search API tem limites mais restritos

        return all_repos

    def fetch_code_files(
        self,
        path: str = "",
        ref: str = "main",
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None,
        max_file_size: int = 1024 * 1024,  # 1MB por padrão
    ) -> List[Dict]:
        """
        Obtém arquivos de código de um repositório, com opções de filtragem avançada.

        Args:
            path: Caminho base dentro do repositório
            ref: Branch ou commit SHA para buscar os arquivos
            recursive: Se deve buscar arquivos em subdiretórios recursivamente
            file_extensions: Lista de extensões de arquivo para filtrar (.py, .js, etc)
            exclude_dirs: Lista de diretórios para excluir da busca
            max_file_size: Tamanho máximo do arquivo em bytes para baixar

        Returns:
            List[Dict]: Lista com todos os arquivos de código e seus conteúdos
        """
        if exclude_dirs is None:
            exclude_dirs = [
                ".git",
                "node_modules",
                "venv",
                "__pycache__",
                "build",
                "dist",
            ]

        if file_extensions is None:
            file_extensions = [
                ".py",
                ".js",
                ".java",
                ".cpp",
                ".h",
                ".c",
                ".ts",
                ".go",
                ".rb",
            ]

        self.logger.info(
            f"🔍 Buscando arquivos de código em {self.owner}/{self.repo}/{path} (ref: {ref})"
        )

        # Obter a estrutura de arquivos usando o endpoint contents
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/contents/{path}"
        params = {"ref": ref}

        contents = self._make_request(
            url,
            params=params,
            error_message=f"Erro ao obter estrutura de arquivos em {path}",
        )

        if not contents:
            self.logger.warning(
                f"❌ Não foi possível obter conteúdo do caminho: {path}"
            )
            return []

        all_files = []

        # Processar cada item no diretório
        for item in contents:
            item_path = item["path"]
            item_type = item["type"]
            item_name = item["name"]

            # Pular diretórios excluídos
            if item_type == "dir" and item_name in exclude_dirs:
                self.logger.debug(f"⏩ Pulando diretório excluído: {item_path}")
                continue

            # Processar subdiretórios recursivamente
            if item_type == "dir" and recursive:
                self.logger.debug(f"📂 Explorando subdiretório: {item_path}")
                subdir_files = self.fetch_code_files(
                    path=item_path,
                    ref=ref,
                    recursive=recursive,
                    file_extensions=file_extensions,
                    exclude_dirs=exclude_dirs,
                    max_file_size=max_file_size,
                )
                all_files.extend(subdir_files)

            # Processar arquivos de código
            elif item_type == "file":
                # Verificar extensão do arquivo
                if file_extensions and not any(
                    item_name.endswith(ext) for ext in file_extensions
                ):
                    self.logger.debug(
                        f"⏩ Arquivo ignorado (extensão não corresponde): {item_path}"
                    )
                    continue

                # Verificar tamanho do arquivo
                if item["size"] > max_file_size:
                    self.logger.warning(
                        f"⏩ Arquivo muito grande, pulando: {item_path} ({item['size']/1024:.1f} KB)"
                    )
                    continue

                # Obter conteúdo do arquivo (já vem em base64 da API do GitHub)
                file_content = self._make_request(
                    item["url"],
                    error_message=f"Erro ao baixar conteúdo do arquivo {item_path}",
                    use_cache=True,
                )

                if file_content and "content" in file_content:
                    try:
                        # O conteúdo vem em base64 e precisa ser decodificado
                        import base64

                        content = base64.b64decode(
                            file_content["content"].replace("\n", "")
                        ).decode("utf-8")

                        file_data = {
                            "path": item_path,
                            "name": item_name,
                            "content": content,
                            "size": item["size"],
                            "sha": item["sha"],
                            "url": item["html_url"],
                        }

                        all_files.append(file_data)
                        self.logger.debug(f"✅ Arquivo processado: {item_path}")
                    except Exception as e:
                        self.logger.error(
                            f"❌ Erro ao decodificar arquivo {item_path}: {str(e)}"
                        )

        self.logger.info(
            f"✅ Total de {len(all_files)} arquivos de código coletados em {path}"
        )
        return all_files
