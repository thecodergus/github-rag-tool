import os
from datetime import datetime
from typing import Optional


def setup_environment():
    """Configura variáveis de ambiente a partir do arquivo .env"""
    from dotenv import load_dotenv

    load_dotenv()

    # Verificar se as chaves necessárias estão configuradas
    required_keys = ["OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        print(
            f"⚠️ As seguintes variáveis de ambiente estão faltando: {', '.join(missing_keys)}"
        )
        return False
    return True


def generate_session_id(prefix: str = "session", use_timestamp: bool = True) -> str:
    """Gera um ID de sessão único"""
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{prefix}_{timestamp}"
    return f"{prefix}_{os.urandom(4).hex()}"


def parse_mongo_connection(connection_string: Optional[str] = None) -> str:
    """Processa string de conexão com MongoDB"""
    if not connection_string:
        connection_string = os.getenv("MONGODB_URI")

    if not connection_string:
        raise ValueError(
            "String de conexão MongoDB não fornecida e MONGODB_URI não está definida"
        )

    return f"mongodb:{connection_string}"
