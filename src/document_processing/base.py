from abc import ABC, abstractmethod
from typing import Dict, List, Any
import hashlib
import json


class DocumentProcessor(ABC):
    """Classe base abstrata para processadores de documentos."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.debug = config.get("debug", False)

    @abstractmethod
    def tokenize(self, content: str) -> List[Dict]:
        """Transforma o conteúdo em tokens estruturados."""
        pass

    @abstractmethod
    def get_format_name(self) -> str:
        """Retorna o nome do formato processado por esta classe."""
        pass

    def create_content_hash(self, content: str) -> str:
        """Cria um hash único para o conteúdo."""
        return hashlib.sha256(content.encode()).hexdigest()[:8]

    def log(self, message: str, data: Any = None):
        """Registra mensagens de log quando o modo debug está ativado."""
        if self.debug:
            print(f"[DEBUG] {message}")
            if data is not None:
                if isinstance(data, (dict, list)):
                    print(
                        json.dumps(data, indent=2, default=str)[:500]
                        + ("..." if len(json.dumps(data)) > 500 else "")
                    )
                else:
                    print(f"  {data}")
