from typing import Dict, Any

class ConfigManager:
    """
    Gerencia as configurações avançadas para GitHubRagTool.
    """
    def __init__(self, initial_config: Dict[str, Any]):
        """
        Inicializa com configurações padrão.
        """
        self.config = initial_config.copy()

    def update(self, options: Dict[str, Any]) -> None:
        """
        Atualiza configurações com base em opções fornecidas.
        """
        self.config.update(options)

    def get(self) -> Dict[str, Any]:
        """
        Retorna as configurações atuais.
        """
        return self.config