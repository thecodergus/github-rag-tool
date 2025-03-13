import os
import uuid
import logging
import time
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union
from contextlib import contextmanager
from functools import wraps

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("github_rag_utils")


def setup_environment(
    env_file: str = ".env",
    required_keys: List[str] = None,
    optional_keys: List[str] = None,
) -> Dict[str, bool]:
    """
    Configura variáveis de ambiente a partir do arquivo .env e valida as chaves necessárias.

    Args:
        env_file: Caminho para o arquivo .env
        required_keys: Lista de chaves que devem estar presentes
        optional_keys: Lista de chaves opcionais a verificar

    Returns:
        Dicionário com o status de cada chave (True se presente, False se ausente)
    """
    from dotenv import load_dotenv

    # Valores padrão
    required_keys = required_keys or ["OPENAI_API_KEY"]
    optional_keys = optional_keys or ["MONGODB_URI", "GITHUB_TOKEN", "LOG_LEVEL"]

    # Carregar variáveis de ambiente
    if os.path.exists(env_file):
        load_dotenv(env_file)
        logger.info(f"Variáveis de ambiente carregadas de {env_file}")
    else:
        logger.warning(
            f"Arquivo {env_file} não encontrado, usando variáveis de ambiente do sistema"
        )

    # Verificar chaves
    env_status = {}

    # Verificar chaves obrigatórias
    for key in required_keys:
        value = os.getenv(key)
        env_status[key] = bool(value)
        if not value:
            logger.error(f"⚠️ Variável de ambiente obrigatória não encontrada: {key}")

    # Verificar chaves opcionais
    for key in optional_keys:
        value = os.getenv(key)
        env_status[key] = bool(value)
        if not value:
            logger.debug(f"Variável de ambiente opcional não encontrada: {key}")

    # Configurar nível de log se especificado
    if log_level := os.getenv("LOG_LEVEL"):
        try:
            numeric_level = getattr(logging, log_level.upper())
            logging.getLogger().setLevel(numeric_level)
            logger.info(f"Nível de log definido para {log_level.upper()}")
        except AttributeError:
            logger.warning(f"Nível de log inválido: {log_level}")

    return env_status


def generate_session_id(
    prefix: str = "session",
    use_timestamp: bool = True,
    include_metadata: Dict[str, str] = None,
    id_format: str = "standard",  # standard, uuid, short
) -> str:
    """
    Gera um ID de sessão único com opções de formato.

    Args:
        prefix: Prefixo para o ID da sessão
        use_timestamp: Se deve incluir timestamp no ID
        include_metadata: Metadados opcionais para incluir no ID
        id_format: Formato do ID (standard, uuid, short)

    Returns:
        ID de sessão formatado
    """
    # Componentes base
    components = [prefix]

    # Adicionar timestamp se solicitado
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        components.append(timestamp)

    # Gerar parte aleatória baseada no formato solicitado
    if id_format == "uuid":
        random_part = str(uuid.uuid4())
    elif id_format == "short":
        random_part = os.urandom(3).hex()
    else:  # standard
        random_part = os.urandom(4).hex()

    components.append(random_part)

    # Adicionar metadados se fornecidos
    if include_metadata:
        for key, value in include_metadata.items():
            if value and key:
                # Garantir que os valores sejam seguros para IDs
                safe_value = re.sub(r"[^a-zA-Z0-9]", "", value)
                if safe_value:
                    components.append(f"{key}-{safe_value}")

    # Juntar componentes
    return "_".join(components)


def parse_mongo_connection(
    connection_string: Optional[str] = None,
    db_name: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Processa e valida string de conexão com MongoDB.

    Args:
        connection_string: URI de conexão MongoDB (opcional)
        db_name: Nome do banco de dados (opcional)
        collection_name: Nome da coleção (opcional)

    Returns:
        Dicionário com as informações de conexão
    """
    # Obter string de conexão
    conn_str = connection_string or os.getenv("MONGODB_URI")

    if not conn_str:
        raise ValueError(
            "String de conexão MongoDB não fornecida e MONGODB_URI não está definida"
        )

    # Verificar formato da string
    if not conn_str.startswith(("mongodb://", "mongodb+srv://")):
        # Adicionar prefixo se estiver faltando
        conn_str = f"mongodb://{conn_str}"

    # Validar formato básico
    pattern = r"^mongodb(\+srv)?://([^:]+:[^@]+@)?([^/]+)(/[^?]+)?(\?.*)?$"
    if not re.match(pattern, conn_str):
        raise ValueError(f"Formato de string de conexão MongoDB inválido: {conn_str}")

    # Extrair e usar db_name se fornecido
    if db_name:
        # Remover database da URI se já estiver presente
        base_uri = re.sub(r"/[^?/]+(\?|$)", "/", conn_str)
        conn_str = f"{base_uri.rstrip('/')}/{db_name}"

    return {"uri": conn_str, "db_name": db_name, "collection_name": collection_name}


@contextmanager
def temp_file_handler(content: str, suffix: str = ".tmp") -> str:
    """
    Context manager para criação e gestão de arquivos temporários.

    Args:
        content: Conteúdo a escrever no arquivo
        suffix: Sufixo para o arquivo temporário

    Yields:
        Caminho para o arquivo temporário
    """
    import tempfile

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        with open(temp.name, "w", encoding="utf-8") as f:
            f.write(content)
        yield temp.name
    finally:
        try:
            os.unlink(temp.name)
        except Exception as e:
            logger.warning(f"Erro ao remover arquivo temporário {temp.name}: {e}")


def timing_decorator(func):
    """
    Decorador para medir o tempo de execução de funções.

    Args:
        func: Função a decorar

    Returns:
        Função decorada que registra tempo de execução
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.debug(f"Função {func.__name__} executada em {duration:.4f} segundos")
        return result

    return wrapper


def format_repository_log(repo_name: str, action: str, details: Dict[str, Any]) -> str:
    """
    Formata logs específicos para operações com repositórios GitHub.

    Args:
        repo_name: Nome do repositório
        action: Ação realizada
        details: Detalhes da operação

    Returns:
        Mensagem formatada
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
    return f"[{timestamp}] [{repo_name}] {action.upper()}: {detail_str}"


def get_memory_usage() -> Dict[str, float]:
    """
    Obtém estatísticas de uso de memória para o processo atual.

    Returns:
        Dicionário com informações de uso de memória
    """
    import psutil
    import os

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return {
        "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size em MB
        "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size em MB
        "percent": process.memory_percent(),
    }


def parse_github_repo_url(url: str) -> Tuple[str, str]:
    """
    Extrai o proprietário (owner) e o nome do repositório a partir de uma URL do GitHub.

    Suporta vários formatos de URL:
    - https://github.com/owner/repo
    - https://github.com/owner/repo.git
    - http://github.com/owner/repo
    - git@github.com:owner/repo.git
    - github.com/owner/repo

    Args:
        url: URL do repositório GitHub

    Returns:
        Tupla (owner, repo_name)

    Raises:
        ValueError: Se a URL não for válida ou não contiver as informações necessárias
    """
    # Remover espaços em branco
    url = url.strip()

    # Padrão para URLs HTTPS/HTTP
    https_pattern = (
        r"(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/\.]+)(?:\.git)?/?$"
    )

    # Padrão para URLs SSH
    ssh_pattern = r"git@github\.com:([^/]+)/([^/\.]+)(?:\.git)?$"

    # Tentar extrair com o padrão HTTPS/HTTP
    match = re.match(https_pattern, url)
    if match:
        owner, repo = match.groups()
        return owner, repo

    # Tentar extrair com o padrão SSH
    match = re.match(ssh_pattern, url)
    if match:
        owner, repo = match.groups()
        return owner, repo

    # Se chegamos aqui, a URL não corresponde a nenhum padrão esperado
    raise ValueError(
        f"URL de repositório inválida: {url}\n"
        "Formatos aceitos:\n"
        "- https://github.com/owner/repo\n"
        "- github.com/owner/repo\n"
        "- git@github.com:owner/repo.git"
    )
