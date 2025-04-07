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

            elif ttype in {"paragraph", "block_code", "list", "block_quote", "table"}:
                # Processa outros tipos que dependem do header_path
                content = token.get("raw", "")

                # Para parágrafos ou elementos com filhos
                if ttype == "paragraph" or "children" in token:
                    content = process_children(token.get("children", []))

                # Tipo de chunk ajustado dinamicamente:
                chunk_type = ttype if ttype != "list" else "list"
                metadata = {}

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
                            header_path=current_hierarchy.copy(),  # Corrigido para usar hierarquia
                            content=content,
                            chunk_type=chunk_type,
                            metadata=metadata,
                            self_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
                        )
                    )

            elif ttype == "thematic_break":
                # Quebras temáticas (ex: "---") ainda estão associadas ao header_path atual
                chunks.append(
                    ContentChunk(
                        header_path=current_hierarchy.copy(),  # Respeita hierarquia do contexto atual
                        content="---",  # Representa o separador
                        chunk_type="thematic_break",
                        metadata={},
                        self_hash=hashlib.sha256("---".encode()).hexdigest()[:16],
                    )
                )

            elif ttype == "blank_line":
                # Linhas em branco não geram conteúdo, apenas mantêm a estrutura
                continue

        # Retorna a estrutura esperada
        return {"children": chunks}

    # HTML Parser
    def _parse_html(self, content: str) -> Dict:
        soup = BeautifulSoup(content, "html.parser")
        return self._build_html_tree(soup.find_all(True))

    def _build_html_tree(self, elements: List, level: int = 0) -> Dict:
        tree = {"children": [], "current_text": ""}
        for element in elements:
            if element.name in self.config["header_levels"]["html"]:
                header_level = int(element.name[1])
                tree["children"].append(
                    {
                        "type": "header",
                        "level": header_level,
                        "text": element.get_text(),
                        "children": self._build_html_tree(
                            element.next_siblings, header_level
                        ),
                    }
                )
            elif element.name == "pre":
                code_content = element.get_text()
                tree["children"].append(
                    ContentChunk(
                        header_path=self._get_current_path(),
                        content=code_content,
                        chunk_type="code_block",
                        metadata={"language": self._detect_code_language(element)},
                    )
                )
            else:
                tree["current_text"] += element.get_text() + "\n"

        # Process remainder text
        if tree["current_text"]:
            tree["children"].append(
                ContentChunk(
                    header_path=self._get_current_path(),
                    content=tree["current_text"],
                    chunk_type="text",
                    metadata={},
                )
            )
        return tree

    # LaTeX Parser
    def _parse_latex(self, content: str) -> Dict:
        sections = re.findall(
            r"\$section|subsection|subsubsection)\*?{(.*?)}", content, re.DOTALL
        )
        structure = []
        current_level = 0
        current_path = []

        for section_type, content in sections:
            level = self.config["header_levels"]["latex"].index(section_type) + 1
            title = content.split("\n")[0].strip()

            if level > current_level:
                current_path.append(title)
            else:
                current_path = current_path[: level - 1] + [title]

            section_content = self._extract_latex_content(content)
            structure.append({"paths": current_path.copy(), "content": section_content})

        return structure

    # PDF Parser
    def _parse_pdf(self, content: bytes) -> Dict:
        doc = pymupdf.open(stream=content, filetype="pdf")
        structure = []
        current_hierarchy = []
        prev_heading = None

        for page in doc:
            blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)[
                "blocks"
            ]

            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if self._is_pdf_heading(span):
                                level = self._determine_pdf_heading_level(span)
                                title = span["text"]

                                if not prev_heading or level <= prev_heading["level"]:
                                    current_hierarchy = current_hierarchy[
                                        : level - 1
                                    ] + [title]

                                structure.append(
                                    {
                                        "paths": current_hierarchy.copy(),
                                        "content": "",
                                        "level": level,
                                    }
                                )
                                prev_heading = {"text": title, "level": level}
                            else:
                                if structure:
                                    structure[-1]["content"] += span["text"]

        return structure

    def _is_pdf_heading(self, span: Dict) -> bool:
        font_size = span["size"]
        font_name = span["font"].lower()
        header_sizes = self.config["header_levels"]["pdf"]["font_sizes"]
        header_fonts = set(
            f.lower() for f in self.config["header_levels"]["pdf"]["font_names"]
        )
        return (font_size in header_sizes) and (font_name in header_fonts)

    def _determine_pdf_heading_level(self, span: Dict) -> int:
        sorted_sizes = sorted(
            set(self.config["header_levels"]["pdf"]["font_sizes"]), reverse=True
        )
        return sorted_sizes.index(span["size"]) + 1


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
