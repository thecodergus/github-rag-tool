import hashlib
import json
from abc import ABC, abstractmethod
from typing import *
from dataclasses import dataclass


@dataclass
class ContentChunk:
    """Representa um chunk de conteúdo com sua hierarquia e metadados."""

    header_path: List[str]
    content: str
    chunk_type: str
    metadata: Dict[str, Any]
    self_hash: str
    parent_hash: Optional[str] = None

    def __repr__(self) -> str:
        """Representação personalizada do objeto ContentChunk."""
        return f"ContentChunk(path={'.'.join(self.header_path)}, type={self.chunk_type}, content={self.content[:30]}...)"


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


class MarkdownProcessor(DocumentProcessor):
    """Processador específico para documentos Markdown."""

    def get_format_name(self) -> str:
        return "markdown"

    def tokenize(self, content: str) -> List[Dict]:
        """
        Transforma o conteúdo Markdown em tokens estruturados.
        """
        try:
            import markdown_it
            from markdown_it.token import Token
        except ImportError:
            raise ImportError(
                "Pacote 'markdown_it' não encontrado. Instale-o com 'pip install markdown-it-py'"
            )

        self.log(
            f"Iniciando tokenização de conteúdo Markdown ({len(content)} caracteres)"
        )

        # Inicializa o parser de markdown com todas as extensões necessárias
        md_parser = markdown_it.MarkdownIt("commonmark", {"html": True})
        md_parser.enable(["table", "strikethrough", "linkify"])

        # Parseia o conteúdo markdown para obter a árvore de tokens
        parsed_tokens = md_parser.parse(content)

        self.log(
            f"Parser gerou {len(parsed_tokens)} tokens brutos",
            [
                {"type": t.type, "tag": t.tag, "content": t.content[:50]}
                for t in parsed_tokens[:5]
            ],
        )

        # Converte os tokens para a estrutura hierárquica
        hierarchical_tokens = self._build_token_hierarchy(parsed_tokens)

        self.log(
            f"Hierarquia de tokens construída com {len(hierarchical_tokens)} tokens raiz"
        )

        # Debug dos tokens hierárquicos
        if self.debug:
            for i, token in enumerate(hierarchical_tokens[:3]):
                self.log(f"Token {i}: {token['type']}", token)

        return hierarchical_tokens

    def _build_token_hierarchy(self, parsed_tokens) -> List[Dict]:
        """
        Constrói uma hierarquia estruturada a partir dos tokens do markdown-it.
        """
        self.log("Construindo hierarquia de tokens")

        hierarchical_tokens = []
        token_stack = []
        current_section = {}

        i = 0
        while i < len(parsed_tokens):
            token = parsed_tokens[i]

            self.log(
                f"Processando token[{i}]: {token.type} (tag={token.tag}, nesting={token.nesting})"
            )

            # Processamento especial para cabeçalhos - corrigindo o problema anterior
            if token.type == "heading_open":
                level = int(token.tag[1])  # Extrai o nível do h1, h2, h3, etc.

                # Avança para o próximo token que deve conter o conteúdo
                i += 1
                if i < len(parsed_tokens) and parsed_tokens[i].type == "inline":
                    header_token = {
                        "type": "heading",
                        "tag": token.tag,
                        "level": level,
                        "content": parsed_tokens[i].content,  # Captura o conteúdo real
                        "children": [],
                        "attrs": {"level": level},
                    }

                    self.log(
                        f"Cabeçalho encontrado: Nível {level}, Conteúdo: '{parsed_tokens[i].content}'"
                    )

                    hierarchical_tokens.append(header_token)

                    # Avança além do token de fechamento
                    i += 2
                    continue

            # Processamento para código cercado (fenced code)
            elif token.type == "fence":
                code_token = {
                    "type": "block_code",
                    "content": token.content,
                    "attrs": {"info": token.info},
                    "marker": token.markup,
                    "children": [],
                }

                self.log(
                    f"Bloco de código encontrado: Linguagem: {token.info}, Conteúdo: '{token.content[:30]}...'"
                )

                hierarchical_tokens.append(code_token)
                i += 1
                continue

            # Processamento para parágrafos
            elif token.type == "paragraph_open":
                # Procura o conteúdo do parágrafo no próximo token
                i += 1
                if i < len(parsed_tokens) and parsed_tokens[i].type == "inline":
                    para_token = {
                        "type": "paragraph",
                        "content": parsed_tokens[i].content,
                        "children": [],
                    }

                    self.log(
                        f"Parágrafo encontrado: '{parsed_tokens[i].content[:30]}...'"
                    )

                    hierarchical_tokens.append(para_token)

                    # Avança além do token de fechamento
                    i += 2
                    continue

            # Para outros tipos de tokens, adicione processamento conforme necessário
            else:
                # Token genérico
                token_dict = {
                    "type": token.type,
                    "tag": token.tag if hasattr(token, "tag") else "",
                    "content": token.content,
                    "children": [],
                }

                if token.content:
                    self.log(
                        f"Token genérico com conteúdo: {token.type}, '{token.content[:30]}...'"
                    )
                    hierarchical_tokens.append(token_dict)

            i += 1

        self.log(f"Hierarquia construída com {len(hierarchical_tokens)} tokens")
        return hierarchical_tokens


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


if __name__ == "__main__":
    import sys
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()

    # Configuração do chunker com modo de depuração ativado
    chunker = HierarchicalChunker(
        {
            "max_chunk_size": 3100,
            "preserve_elements": {"math": False, "code_blocks": True},
            "debug": True,  # Ativa depuração
        }
    )

    # Processa o arquivo Markdown
    try:
        with open("test_document.md", "r", encoding="utf-8") as f:
            content = f.read()
            console.print(
                Panel(
                    f"[bold green]Arquivo carregado com sucesso[/bold green]\n{len(content)} caracteres",
                    title="Status",
                )
            )
    except FileNotFoundError:
        console.print(
            Panel(
                "[bold red]Arquivo test_document.md não encontrado![/bold red]",
                title="Erro",
            )
        )
        sys.exit(1)

    # Processa o conteúdo
    console.print("[bold]Processando documento Markdown...[/bold]")
    chunks_md = chunker.chunk(content, "markdown")
    console.print(
        f"[bold green]✓[/bold green] [bold]{len(chunks_md)} chunks gerados[/bold]"
    )

    # Exibe os chunks resultantes
    console.print("\n[bold]Lista de Chunks Processados:[/bold]")
    for i, chunk in enumerate(chunks_md, 1):
        # Formata o caminho de cabeçalhos
        header_path = " > ".join(chunk.header_path) if chunk.header_path else "Raiz"

        # Cria texto enriquecido para o conteúdo
        chunk_text = Text.from_markup(f"[bold]{chunk.chunk_type.upper()}[/bold]")

        # Adiciona detalhes baseados no tipo de chunk
        if chunk.chunk_type == "heading":
            title = f"H{chunk.metadata.get('level', '?')} - {chunk.content}"
            color = f"rgb({180-chunk.metadata.get('level', 1)*30},{100+chunk.metadata.get('level', 1)*20},255)"
            chunk_text = Text.from_markup(f"[bold {color}]{title}[/bold {color}]")
        elif chunk.chunk_type == "block_code":
            lang = chunk.metadata.get("language", "")
            lang_info = f" ({lang})" if lang else ""
            chunk_text = Text.from_markup(f"[bold cyan]CÓDIGO{lang_info}[/bold cyan]")

        # Cria e exibe o painel para o chunk
        metadata_str = ", ".join(
            [
                f"{k}: {v}"
                for k, v in chunk.metadata.items()
                if k not in ["section", "level"]
            ]
        )
        panel_subtitle = (
            f"Hash: {chunk.self_hash} | Pai: {chunk.parent_hash or 'Nenhum'}"
        )

        panel = Panel(
            Text(
                chunk.content[:100] + "..."
                if len(chunk.content) > 100
                else chunk.content
            ),
            title=chunk_text,
            subtitle=panel_subtitle,
            subtitle_align="right",
            border_style=f"{'green' if i % 2 == 0 else 'blue'}",
        )
        console.print(f"[{i}] [bold yellow]{header_path}[/bold yellow]")
        console.print(panel)

        # Adiciona separadores entre chunks
        if i < len(chunks_md):
            console.print("─" * 80)
