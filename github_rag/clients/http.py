import requests
from typing import Any, Dict, Optional

class HTTPRequestManager:
    """
    Gerencia requisições HTTP usando uma sessão compartilhada.
    """
    def __init__(self):
        self.session = requests.Session()

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        timeout: int = 30,
    ) -> requests.Response:
        """
        Executa uma requisição HTTP e retorna o objeto Response.

        Args:
            method: Método HTTP ('GET', 'POST', etc).
            url: URL da requisição.
            headers: Cabeçalhos HTTP.
            params: Parâmetros de query string.
            data: Corpo da requisição (serializado como JSON).
            timeout: Tempo máximo de espera em segundos.
        """
        return self.session.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=data,
            timeout=timeout,
        )