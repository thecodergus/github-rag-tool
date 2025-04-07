from .base import DocumentProcessor
from typing import Dict, List


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
