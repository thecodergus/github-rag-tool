import os
import time
import json
import hashlib
import requests
from typing import Dict, List, Optional, Tuple, Any, Union


class GitHubClient:
    """
    Cliente para interação com a API do GitHub com tratamento avançado de limites de taxa,
    cache de requisições e backoff exponencial.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: str = ".github_cache",
        cache_ttl: int = 86400,  # 24 horas em segundos
    ):
        self.base_url = "https://api.github.com"
        self.headers = {"Accept": "application/vnd.github.v3+json"}

        # Configuração de autenticação
        if token:
            self.headers["Authorization"] = f"token {token}"

        # Configuração de cache
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl

        if use_cache and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Estatísticas de uso da API
        self.requests_made = 0
        self.cache_hits = 0
        self.rate_limit_hits = 0

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
            if time.time() - os.path.getmtime(cache_file) < self.cache_ttl:
                try:
                    with open(cache_file, "r") as f:
                        self.cache_hits += 1
                        return json.load(f)
                except Exception as e:
                    print(f"⚠️ Erro ao ler cache: {str(e)}")

        return None

    def _save_to_cache(self, cache_key: str, data: Dict) -> None:
        """Salva dados no cache."""
        if not self.use_cache:
            return

        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"⚠️ Erro ao salvar cache: {str(e)}")

    def check_rate_limit(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Verifica os limites de taxa atuais da API.

        Returns:
            Tuple[int, int]: (requisições restantes, timestamp para reset)
        """
        url = f"{self.base_url}/rate_limit"

        try:
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                data = response.json()

                # Obter limites de taxa core e search
                core_rate = data["resources"]["core"]
                search_rate = data["resources"]["search"]

                # Informar limites de taxa
                print("\n--- Limites de Taxa da API GitHub ---")
                print(
                    f"📊 Core API: {core_rate['remaining']}/{core_rate['limit']} restantes"
                )
                print(
                    f"🔎 Search API: {search_rate['remaining']}/{search_rate['limit']} restantes"
                )

                # Calcular tempo até reset
                core_reset = time.strftime(
                    "%H:%M:%S", time.localtime(core_rate["reset"])
                )
                search_reset = time.strftime(
                    "%H:%M:%S", time.localtime(search_rate["reset"])
                )

                print(f"⏱️ Core API reset às: {core_reset}")
                print(f"⏱️ Search API reset às: {search_reset}")

                # Avisar se estiver próximo do limite
                if core_rate["remaining"] < (core_rate["limit"] * 0.1):
                    print(f"⚠️ ATENÇÃO: Menos de 10% das requisições Core disponíveis!")

                return core_rate["remaining"], core_rate["reset"]
            else:
                print(f"❌ Erro ao verificar limites de taxa: {response.status_code}")
                return None, None

        except Exception as e:
            print(f"❌ Exceção ao verificar limites de taxa: {str(e)}")
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
                print(f"🔄 Usando dados em cache para: {url}")
                return cached_data

        # Fazer a requisição com retentativas
        retries = 0
        self.requests_made += 1

        while retries < max_retries:
            try:
                if method == "GET":
                    response = requests.get(url, headers=self.headers, params=params)
                elif method == "POST":
                    response = requests.post(
                        url, headers=self.headers, params=params, json=data
                    )
                elif method == "PUT":
                    response = requests.put(
                        url, headers=self.headers, params=params, json=data
                    )
                elif method == "DELETE":
                    response = requests.delete(url, headers=self.headers, params=params)
                else:
                    raise ValueError(f"Método HTTP não suportado: {method}")

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
                        reset_datetime = time.strftime(
                            "%H:%M:%S", time.localtime(reset_time)
                        )
                        print(
                            f"⚠️ Apenas {remaining}/{limit} requisições restantes até {reset_datetime}"
                        )

                    return result

                # Tratar limites de taxa (403/429)
                elif response.status_code in (403, 429):
                    self.rate_limit_hits += 1

                    # Verificar se é realmente um problema de limite de taxa
                    if remaining == 0 and reset_time > 0:
                        current_time = time.time()
                        sleep_time = (
                            max(reset_time - current_time, 0) + 2
                        )  # Margem de segurança

                        print(
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
                        print(
                            f"⏳ Taxa excedida. Aguardando {retry_after} segundos conforme solicitado."
                        )
                        time.sleep(retry_after)
                        retries += 1
                        continue

                    # Backoff exponencial para outras tentativas
                    wait_time = (2**retries) + (
                        time.time() % 1
                    )  # Adiciona um pouco de aleatoriedade
                    print(
                        f"⏳ {error_message} ({response.status_code}). Tentativa {retries+1}/{max_retries} em {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                    retries += 1

                # Outros erros
                else:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get("message", "Sem detalhes do erro")
                    print(f"❌ {error_message}: {response.status_code} - {error_msg}")

                    # Alguns erros não devem ser retentados
                    if response.status_code in (401, 404, 422):
                        return None

                    # Para outros erros, tentar novamente com backoff
                    wait_time = (2**retries) + (time.time() % 1)
                    print(f"⏳ Tentativa {retries+1}/{max_retries} em {wait_time:.1f}s")
                    time.sleep(wait_time)
                    retries += 1

            except requests.exceptions.RequestException as e:
                print(f"❌ Erro de conexão: {str(e)}")
                wait_time = (2**retries) + (time.time() % 1)
                print(f"⏳ Tentativa {retries+1}/{max_retries} em {wait_time:.1f}s")
                time.sleep(wait_time)
                retries += 1

        # Se chegou aqui, todas as tentativas falharam
        print(f"❌ Falha após {max_retries} tentativas para: {url}")
        return None

    def get_user(self, username: str) -> Optional[Dict]:
        """Obtém informações de um usuário."""
        url = f"{self.base_url}/users/{username}"
        return self._make_request(
            url, error_message=f"Erro ao obter usuário {username}"
        )

    def get_repository(self, owner: str, repo: str) -> Optional[Dict]:
        """Obtém informações de um repositório."""
        url = f"{self.base_url}/repos/{owner}/{repo}"
        return self._make_request(
            url, error_message=f"Erro ao obter repositório {owner}/{repo}"
        )

    def get_issues(
        self, owner: str, repo: str, state: str = "all", per_page: int = 100
    ) -> List[Dict]:
        """
        Obtém todas as issues de um repositório, lidando com paginação.

        Args:
            owner: Dono do repositório
            repo: Nome do repositório
            state: Estado das issues (open, closed, all)
            per_page: Número de itens por página

        Returns:
            List[Dict]: Lista com todas as issues
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/issues"
        params = {"state": state, "per_page": per_page, "page": 1}

        all_issues = []

        while True:
            issues = self._make_request(
                url,
                params=params,
                error_message=f"Erro ao obter issues da página {params['page']}",
            )

            if not issues or len(issues) == 0:
                break

            all_issues.extend(issues)

            # Verificar se tem mais páginas
            if len(issues) < per_page:
                break

            params["page"] += 1

            # Pequena pausa entre requisições para evitar sobrecarga
            time.sleep(0.25)

        return all_issues

    def get_pull_requests(
        self, owner: str, repo: str, state: str = "all", per_page: int = 100
    ) -> List[Dict]:
        """
        Obtém todos os pull requests de um repositório, lidando com paginação.

        Args:
            owner: Dono do repositório
            repo: Nome do repositório
            state: Estado dos PRs (open, closed, all)
            per_page: Número de itens por página

        Returns:
            List[Dict]: Lista com todos os pull requests
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
        params = {"state": state, "per_page": per_page, "page": 1}

        all_prs = []

        while True:
            prs = self._make_request(
                url,
                params=params,
                error_message=f"Erro ao obter PRs da página {params['page']}",
            )

            if not prs or len(prs) == 0:
                break

            all_prs.extend(prs)

            # Verificar se tem mais páginas
            if len(prs) < per_page:
                break

            params["page"] += 1

            # Pequena pausa entre requisições para evitar sobrecarga
            time.sleep(0.25)

        return all_prs

    def get_commits(
        self, owner: str, repo: str, per_page: int = 100, since: Optional[str] = None
    ) -> List[Dict]:
        """
        Obtém todos os commits de um repositório, lidando com paginação.

        Args:
            owner: Dono do repositório
            repo: Nome do repositório
            per_page: Número de itens por página
            since: Data ISO 8601 para filtrar commits (YYYY-MM-DDTHH:MM:SSZ)

        Returns:
            List[Dict]: Lista com todos os commits
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/commits"
        params = {"per_page": per_page, "page": 1}

        if since:
            params["since"] = since

        all_commits = []

        while True:
            commits = self._make_request(
                url,
                params=params,
                error_message=f"Erro ao obter commits da página {params['page']}",
            )

            if not commits or len(commits) == 0:
                break

            all_commits.extend(commits)

            # Verificar se tem mais páginas
            if len(commits) < per_page:
                break

            params["page"] += 1

            # Pequena pausa entre requisições para evitar sobrecarga
            time.sleep(0.25)

        return all_commits

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
                print(f"📊 Total de resultados encontrados: {total_count}")

            items = result["items"]
            all_repos.extend(items)

            # Verificar se tem mais páginas
            if len(items) < per_page or len(all_repos) >= total_count:
                break

            params["page"] += 1

            # Pequena pausa entre requisições para evitar sobrecarga
            time.sleep(0.5)  # Search API tem limites mais restritos

        return all_repos

    def get_statistics(self) -> Dict[str, int]:
        """Retorna estatísticas de uso do cliente."""
        return {
            "requests_made": self.requests_made,
            "cache_hits": self.cache_hits,
            "rate_limit_hits": self.rate_limit_hits,
        }

    def clear_cache(self) -> None:
        """Limpa todo o cache de requisições."""
        if not self.use_cache:
            return

        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".json"):
                    os.remove(os.path.join(self.cache_dir, filename))
            print(f"✅ Cache limpo com sucesso.")
        except Exception as e:
            print(f"❌ Erro ao limpar cache: {str(e)}")

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
        # Verifica se já atingimos o limite de arquivos
        if max_files is not None and max_files <= 0:
            return []

        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/contents/{path}"

        try:
            print(f"🔍 Explorando diretório: {path or 'raiz'}")

            # Usa o método _make_request que já tem tratamento de limites de taxa e cache
            contents = self._make_request(
                url, error_message=f"Falha ao acessar {path}", retries=3
            )

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
                    try:
                        # Verifica se já atingimos o limite de arquivos
                        if max_files is not None and len(files) >= max_files:
                            print(f"\n✅ Limite de {max_files} arquivos atingido.")
                            return files

                        print(f"📄 Baixando: {item['path']}")

                        # Usa o sistema de cache para evitar downloads repetidos
                        cache_key = self._get_cache_key(item["download_url"])
                        content = self._get_from_cache(cache_key)

                        if content is None:
                            # Se não estiver em cache, baixa o conteúdo respeitando limites de taxa
                            content = self._make_request(
                                item["download_url"],
                                use_base_url=False,
                                error_message=f"Falha ao baixar {item['path']}",
                                retries=2,
                            )

                            # Salva no cache para uso futuro
                            if content is not None:
                                self._save_to_cache(cache_key, content)
                        else:
                            print(f"📋 Usando versão em cache para: {item['path']}")

                        # Adiciona à lista de arquivos
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

                    except Exception as e:
                        print(
                            f"⚠️ Problema ao processar arquivo {item['path']}: {str(e)}"
                        )

                elif item["type"] == "dir":
                    dirs.append(item["path"])

                # Pequena pausa adaptativa para evitar atingir limites de taxa
                remaining, _ = self._get_rate_limit_from_headers()
                if (
                    remaining and remaining < 100
                ):  # Se estiver com poucas requisições disponíveis
                    time.sleep(1.0)  # Pausa maior
                else:
                    time.sleep(0.3)  # Pausa normal reduzida

            if path:
                print()  # Nova linha após terminar o processamento do diretório

            # Calcula quantos arquivos ainda podemos buscar se há um limite
            remaining_files = None
            if max_files is not None:
                remaining_files = max_files - len(files)
                if remaining_files <= 0:
                    return files

            # Recursivamente buscar em subdiretórios
            for dir_path in dirs:
                subdir_files = self.fetch_code_files(
                    dir_path, max_files=remaining_files
                )
                files.extend(subdir_files)

                # Atualiza o contador de arquivos restantes
                if remaining_files is not None:
                    remaining_files -= len(subdir_files)
                    if remaining_files <= 0:
                        break

            return files

        except Exception as e:
            print(f"❌ Erro ao processar diretório {path}: {str(e)}")
            # Verifica se é um erro de limite de taxa e aguarda se necessário
            if "rate limit exceeded" in str(e).lower():
                self.rate_limit_hits += 1
                print("⏱️ Limite de taxa atingido. Aguardando...")
                self._handle_rate_limit()
                # Tenta novamente após aguardar
                return self.fetch_code_files(path, max_files)

            return []

    def _is_code_file(self, filename: str) -> bool:
        """
        Verifica se um arquivo deve ser considerado para análise.
        Aceita qualquer arquivo de texto que não seja log.
        Ignora arquivos binários, imagens, vídeos e outras mídias.

        Args:
            filename: Nome do arquivo a ser verificado

        Returns:
            True se o arquivo deve ser incluído, False caso contrário
        """

        # Extensões de arquivos a serem ignorados (binários, mídia, etc.)
        ignored_extensions = [
            # Imagens
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".tiff",
            ".webp",
            ".svg",
            ".ico",
            # Vídeos
            ".mp4",
            ".avi",
            ".mov",
            ".wmv",
            ".flv",
            ".mkv",
            ".webm",
            ".m4v",
            # Áudio
            ".mp3",
            ".wav",
            ".ogg",
            ".flac",
            ".aac",
            ".m4a",
            # Documentos binários
            ".pdf",
            ".doc",
            ".docx",
            ".ppt",
            ".pptx",
            ".xls",
            ".xlsx",
            ".odt",
            # Arquivos compactados
            ".zip",
            ".tar",
            ".gz",
            ".rar",
            ".7z",
            ".bz2",
            ".xz",
            # Executáveis e binários
            ".exe",
            ".dll",
            ".so",
            ".dylib",
            ".class",
            ".pyc",
            ".pyd",
            ".o",
            ".obj",
            # Outros binários
            ".bin",
            ".dat",
            ".db",
            ".sqlite",
            ".sqlite3",
            ".mdb",
            ".pkl",
            ".parquet",
        ]

        # Extensões de arquivos de log
        log_extensions = [".log", ".logs", ".logfile"]

        filename_lower = filename.lower()

        # Verificações em ordem de prioridade

        # 1. Se for um arquivo de log, ignora
        if (
            any(filename_lower.endswith(ext) for ext in log_extensions)
            or "log" in filename_lower
        ):
            return False

        # 2. Se tiver uma extensão ignorada, ignora
        if any(filename_lower.endswith(ext) for ext in ignored_extensions):
            return False

        # 3. Arquivos sem extensão ou com extensões desconhecidas
        # Ignora arquivos sem extensão pois podem ser binários
        if "." not in filename_lower:
            return False

        # 4. Para outros casos, assumimos que é um arquivo de texto
        # que pode ser útil para análise
        return True

    def _get_rate_limit_from_headers(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Extrai informações de limite de taxa dos cabeçalhos da última resposta.

        Returns:
            Tuple(remaining, reset): Requisições restantes e timestamp para reset
        """
        if not hasattr(self, "_last_response") or not self._last_response:
            return None, None

        headers = self._last_response.headers

        # Extrai informações de limite de taxa
        remaining = headers.get("X-RateLimit-Remaining")
        reset = headers.get("X-RateLimit-Reset")

        if remaining is not None:
            remaining = int(remaining)

        if reset is not None:
            reset = int(reset)

        return remaining, reset

    def _handle_rate_limit(self):
        """
        Lida com situações de limite de taxa atingido.
        Aguarda até que o limite seja restaurado.
        """
        _, reset_time = self.check_rate_limit()

        if reset_time:
            current_time = int(time.time())
            wait_time = max(reset_time - current_time + 5, 10)  # +5 segundos de margem

            print(
                f"⏱️ Aguardando {wait_time} segundos para o reset do limite de taxa..."
            )

            # Feedback visual da espera
            for i in range(wait_time):
                time_left = wait_time - i
                print(f"⏳ Tempo restante: {time_left}s", end="\r")
                time.sleep(1)

            print("\n✅ Limite de taxa restaurado. Continuando operação...")
        else:
            # Se não conseguir obter o tempo de reset, aguarda um tempo padrão
            print("⏱️ Aguardando 60 segundos antes de tentar novamente...")
            time.sleep(60)
