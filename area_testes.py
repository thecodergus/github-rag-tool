import dataclasses
import hashlib
from typing import *
import re
from bs4 import BeautifulSoup
import pymupdf
from lxml import etree
import mistune
from dataclasses import dataclass


@dataclass
class ContentChunk:
    header_path: List[str]
    content: str
    chunk_type: str
    metadata: Dict
    parent_hash: str = ""
    self_hash: str = ""


@dataclass
class StackFrame:
    node: Dict
    current_path: List[str]
    current_level: int
    parent_chunk: Optional[ContentChunk] = None


class HierarchicalChunker:
    def __init__(self, config: Dict):
        self.config = {
            "max_chunk_size": 3100,
            "header_levels": {
                "markdown": ["#", "##", "###", "####", "#####"],
                "html": ["h1", "h2", "h3", "h4"],
                "latex": ["section", "subsection", "subsubsection", "subsubsubsection"],
                "pdf": {
                    "font_sizes": [20, 18, 16, 14],
                    "font_names": ["Helv", "Arial"],
                },
            },
            "preserve_elements": {"code_blocks": True, "tables": True, "math": True},
        }
        self.config.update(config)

    def chunk(self, content: str, file_type: str) -> List[ContentChunk]:
        parser = self._get_parser(file_type)
        tree = parser(content)

        return tree["children"]

    def _get_parser(self, file_type: str):
        return {
            "markdown": self._parse_markdown,
            "html": self._parse_html,
            "latex": self._parse_latex,
            "pdf": self._parse_pdf,
        }[file_type]

    def _get_current_path(self) -> str:
        """
        Constrói e retorna o caminho hierárquico atual baseado nos chunks ativos.

        Returns:
            str: Caminho completo separado por pontos, ex: 'root.section1.subsection'
        """
        # Filtra chunks vazios e junta com pontos
        active_chunks = [chunk for chunk in self.chunks if chunk.strip()]
        return ".".join(active_chunks) if active_chunks else "root"

    def _parse_markdown(self, content: str) -> Dict:
        """
        Processa conteúdo Markdown usando o AST (Abstract Syntax Tree) do Mistune.
        Retorna uma estrutura de dicionário com chunks processados.
        """
        chunks = []
        current_hierarchy = []

        # Configuração do parser Mistune com AST renderer
        markdown = mistune.create_markdown(renderer="ast")

        # Extrai apenas os tokens AST da tupla retornada
        tokens_ast = markdown(content)

        def process_children(children: List[Dict]) -> str:
            """Processa recursivamente os tokens filhos para extrair texto."""
            text_parts = []
            for child in children:
                if child["type"] == "text":
                    text_parts.append(child.get("raw", ""))
                elif child["type"] == "emphasis":
                    inside = process_children(child["children"])
                    text_parts.append(f"*{inside}*")
                elif child["type"] == "strong":
                    inside = process_children(child["children"])
                    text_parts.append(f"**{inside}**")
                elif child["type"] == "link":
                    text = process_children(child["children"])
                    url = child["attrs"].get("url", "#")
                    text_parts.append(f"[{text}]({url})")
                elif child["type"] == "image":
                    alt = process_children(child["children"])
                    url = child["attrs"].get("url", "#")
                    text_parts.append(f"![{alt}]({url})")
                elif child["type"] == "codespan":
                    text_parts.append(f"`{child.get('raw', '')}`")
                elif child["type"] == "linebreak":
                    text_parts.append("\n")
                elif child["type"] == "softbreak":
                    text_parts.append(" ")
            return "".join(text_parts)

        def handle_list(list_token: Dict, indent: int = 0) -> str:
            """Processa tokens de lista (ordenada ou não-ordenada)."""
            is_ordered = list_token["attrs"].get("ordered", False)
            items = []
            for idx, item in enumerate(list_token.get("children", []), 1):
                # Processa o texto do item da lista
                item_text = ""
                for child in item.get("children", []):
                    if child["type"] == "block_text":
                        item_text += process_children(child["children"])
                    elif child["type"] == "list":
                        # Lista aninhada
                        item_text += "\n" + handle_list(child, indent + 1)
                    elif child["type"] == "block_code":
                        # Código dentro de item de lista
                        code_content = child.get("raw", "")
                        language = child["attrs"].get("info", "")
                        chunks.append(
                            ContentChunk(
                                header_path=current_hierarchy.copy(),
                                content=code_content,
                                chunk_type="code_block",
                                metadata={"language": language},
                                self_hash=hashlib.sha256(
                                    code_content.encode()
                                ).hexdigest()[:16],
                            )
                        )

                # Formata o item com indentação apropriada
                prefix = " " * (indent * 2)
                if is_ordered:
                    items.append(f"{prefix}{idx}. {item_text}")
                else:
                    items.append(f"{prefix}- {item_text}")

            return "\n".join(items)

        # Variável para rastrear o último hash de cabeçalho
        last_header_hash = ""

        for token in tokens_ast:
            ttype = token.get("type", "")

            if ttype == "heading":
                # Processa cabeçalhos e atualiza hierarquia
                level = token["attrs"]["level"]
                header_text = process_children(token["children"])

                if level > len(current_hierarchy):
                    current_hierarchy.append(header_text)
                else:
                    current_hierarchy = current_hierarchy[: level - 1] + [header_text]

                # Atualiza o hash do último cabeçalho
                last_header_hash = hashlib.sha256(header_text.encode()).hexdigest()[:16]

            elif ttype in {"paragraph", "block_code", "list", "block_quote", "table"}:
                # Processa outros tipos que dependem do header_path
                content = token.get("raw", "")

                # Para parágrafos ou elementos com filhos
                if ttype == "paragraph" or "children" in token:
                    content = process_children(token.get("children", []))

                # Tipo de chunk ajustado dinamicamente:
                chunk_type = ttype if ttype != "list" else "list"
                metadata = {
                    "section": (
                        ".".join(current_hierarchy) if current_hierarchy else "root"
                    ),
                    "level": len(current_hierarchy),
                }

                # Adiciona metadata para listas e blocos de código
                if ttype == "list":
                    metadata.update(
                        {
                            "ordered": token["attrs"].get("ordered", False),
                            "depth": token["attrs"].get("depth", 0),
                        }
                    )
                elif ttype == "block_code":
                    metadata.update(
                        {
                            "language": token["attrs"].get("info", ""),
                            "fence": token.get("marker", "```"),
                        }
                    )

                if content.strip():
                    chunks.append(
                        ContentChunk(
                            header_path=current_hierarchy.copy(),
                            content=content,
                            chunk_type=chunk_type,
                            metadata=metadata,
                            parent_hash=last_header_hash,  # Adiciona referência ao último cabeçalho
                            self_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
                        )
                    )

            elif ttype == "thematic_break":
                # Quebras temáticas (ex: "---") ainda estão associadas ao header_path atual
                chunks.append(
                    ContentChunk(
                        header_path=current_hierarchy.copy(),
                        content="---",
                        chunk_type="thematic_break",
                        metadata={
                            "section": (
                                ".".join(current_hierarchy)
                                if current_hierarchy
                                else "root"
                            )
                        },
                        parent_hash=last_header_hash,  # Adiciona referência ao último cabeçalho
                        self_hash=hashlib.sha256("---".encode()).hexdigest()[:16],
                    )
                )

            elif ttype == "blank_line":
                # Linhas em branco não geram conteúdo, apenas mantêm a estrutura
                continue

        # Retorna a estrutura esperada
        return {"children": chunks}


if __name__ == "__main__":
    chunker = HierarchicalChunker(
        {
            "max_chunk_size": 3100,
            "preserve_elements": {"math": False, "code_blocks": True},
        }
    )

    # Para Markdown
    with open("test_document.md", "r") as f:
        chunks_md = chunker.chunk(f.read(), "markdown")

    for m in chunks_md:
        print(m)
    # with open("test_document.md", "r") as f:
    #     chunks_html = chunker.chunk(f.read(), "html")

    # print(chunks_html)
