import hashlib
import json
from typing import Dict, List, Any

from area_testes import MarkdownProcessor
from ..models.content_chunk import ContentChunk


class HierarchicalChunker:
    """Processa documentos em diferentes formatos e os divide em chunks hierárquicos."""

    def __init__(self, config: Dict):
        self.config = config.copy()
        self.debug = config.get("debug", False)

        self.processors = {"markdown": MarkdownProcessor(self.config)}

        # Estado mantido durante o processamento
        self.chunks = []
        self.current_hierarchy = []
        self.last_header_hash = None

    def log(self, message: str, data: Any = None):
        """Registra mensagens de log quando o modo debug está ativado."""
        if self.debug:
            print(f"[CHUNKER] {message}")
            if data is not None:
                if isinstance(data, (dict, list)):
                    print(
                        json.dumps(data, indent=2, default=str)[:300]
                        + ("..." if len(json.dumps(data)) > 300 else "")
                    )
                else:
                    print(f"  {data}")

    def chunk(self, content: str, format_type: str) -> List[ContentChunk]:
        """
        Processa o conteúdo e o divide em chunks baseados na hierarquia.
        """
        self.chunks = []
        self.current_hierarchy = []
        self.last_header_hash = None

        self.log(
            f"Iniciando chunking de documento {format_type} ({len(content)} caracteres)"
        )

        # Obtém o processador adequado para o formato
        format_type = format_type.lower()
        if format_type not in self.processors:
            raise ValueError(
                f"Formato não suportado: {format_type}. Formatos disponíveis: {list(self.processors.keys())}"
            )

        processor = self.processors[format_type]

        # Tokeniza o conteúdo usando o processador específico
        tokens = processor.tokenize(content)

        self.log(f"Documento tokenizado. Processando {len(tokens)} tokens principais")

        # Processa os tokens para gerar os chunks
        self._process_tokens(tokens)

        self.log(f"Processamento concluído. Gerados {len(self.chunks)} chunks")

        return self.chunks

    def _process_tokens(self, tokens: List[Dict]) -> None:
        """Processa a lista de tokens e gera chunks com base na hierarquia."""
        for idx, token in enumerate(tokens):
            ttype = token.get("type", "")
            self.log(f"Processando token {idx}: {ttype}")

            if ttype == "heading":
                self._process_heading_token(token)
            elif ttype == "paragraph":
                self._process_paragraph_token(token)
            elif ttype == "block_code":
                self._process_code_block_token(token)
            elif ttype == "thematic_break":
                self._process_thematic_break_token(token)
            elif ttype == "blank_line":
                continue  # Ignora linhas em branco
            else:
                self._process_generic_token(token)

    def _process_heading_token(self, token: Dict) -> None:
        """Processa tokens de cabeçalho e atualiza a hierarquia."""
        level = token.get("level", token.get("attrs", {}).get("level", 1))
        header_text = token.get("content", "")

        self.log(f"Processando cabeçalho nível {level}: '{header_text}'")

        # Atualiza a hierarquia
        if level <= len(self.current_hierarchy):
            self.current_hierarchy = self.current_hierarchy[: level - 1]

        self.current_hierarchy.append(header_text)

        # Atualiza o hash do último cabeçalho
        content_for_hash = header_text or "empty_header"
        self.last_header_hash = self._create_content_hash(content_for_hash)

        # Cria um chunk para o cabeçalho
        self._add_chunk(
            content=header_text,
            chunk_type="heading",
            metadata={"section": self._get_current_path(), "level": level},
        )

    def _process_paragraph_token(self, token: Dict) -> None:
        """Processa tokens de parágrafo."""
        content = token.get("content", "")

        self.log(f"Processando parágrafo: '{content[:50]}...'")

        if content.strip():
            self._add_chunk(
                content=content,
                chunk_type="paragraph",
                metadata={
                    "section": self._get_current_path(),
                    "level": len(self.current_hierarchy),
                },
            )

    def _process_code_block_token(self, token: Dict) -> None:
        """Processa tokens de bloco de código."""
        content = token.get("content", "")
        language = token.get("attrs", {}).get("info", "")

        self.log(f"Processando bloco de código ({language}): '{content[:50]}...'")

        self._add_chunk(
            content=content,
            chunk_type="block_code",
            metadata={
                "section": self._get_current_path(),
                "level": len(self.current_hierarchy),
                "language": language,
                "fence": token.get("marker", "```"),
            },
        )

    def _process_thematic_break_token(self, token: Dict) -> None:
        """Processa tokens de quebra temática."""
        self._add_chunk(
            content="---",
            chunk_type="thematic_break",
            metadata={"section": self._get_current_path()},
        )

    def _process_generic_token(self, token: Dict) -> None:
        """Processa outros tipos de tokens."""
        content = token.get("content", "")

        if content.strip():
            self.log(
                f"Processando token genérico ({token.get('type')}): '{content[:50]}...'"
            )

            self._add_chunk(
                content=content,
                chunk_type=token.get("type", "unknown"),
                metadata={
                    "section": self._get_current_path(),
                    "level": len(self.current_hierarchy),
                },
            )

    def _get_current_path(self) -> str:
        """Constrói e retorna o caminho hierárquico atual baseado nos chunks ativos."""
        return ".".join(self.current_hierarchy) if self.current_hierarchy else "root"

    def _create_content_hash(self, content: str) -> str:
        """Cria um hash único para o conteúdo."""
        return hashlib.sha256(content.encode()).hexdigest()[:8]

    def _add_chunk(
        self, content: str, chunk_type: str, metadata: Dict[str, Any]
    ) -> None:
        """Adiciona um novo chunk à lista de chunks."""
        content_hash = self._create_content_hash(content)

        self.log(
            f"Adicionando chunk tipo {chunk_type}: Hash={content_hash}, Pai={self.last_header_hash or 'None'}"
        )

        self.chunks.append(
            ContentChunk(
                header_path=self.current_hierarchy.copy(),
                content=content,
                chunk_type=chunk_type,
                metadata=metadata,
                parent_hash=self.last_header_hash,
                self_hash=content_hash,
            )
        )
