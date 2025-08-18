"""
Sistema de chunking inteligente para diferentes tipos de conteúdo.
Implementa estratégias específicas para código, documentação e issues.
"""
import ast
import re
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language


class ChunkingStrategy(ABC):
    """Estratégia abstrata para chunking de diferentes tipos de conteúdo."""
    
    @abstractmethod
    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Divide o conteúdo em chunks inteligentes."""
        pass


class CodeChunkingStrategy(ChunkingStrategy):
    """Estratégia de chunking especializada para código."""
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Mapeamento de extensões para linguagens suportadas pelo LangChain
        self.language_map = {
            # Python
            '.py': Language.PYTHON,
            '.pyw': Language.PYTHON,
            '.pyi': Language.PYTHON,
            
            # JavaScript/TypeScript
            '.js': Language.JS,
            '.mjs': Language.JS,
            '.jsx': Language.JS,
            '.ts': Language.TS,
            '.tsx': Language.TS,
            
            # Java/JVM Languages
            '.java': Language.JAVA,
            '.kt': Language.KOTLIN,
            '.kts': Language.KOTLIN,
            '.scala': Language.SCALA,
            '.sc': Language.SCALA,
            
            # C/C++
            '.c': Language.C,
            '.cpp': Language.CPP,
            '.cxx': Language.CPP,
            '.cc': Language.CPP,
            '.c++': Language.CPP,
            '.hpp': Language.CPP,
            '.hxx': Language.CPP,
            '.h': Language.C,
            '.hh': Language.CPP,
            
            # C#
            '.cs': Language.CSHARP,
            '.csx': Language.CSHARP,
            
            # Other Languages
            '.go': Language.GO,
            '.rs': Language.RUST,
            '.php': Language.PHP,
            '.php3': Language.PHP,
            '.php4': Language.PHP,
            '.php5': Language.PHP,
            '.phtml': Language.PHP,
            '.rb': Language.RUBY,
            '.rbw': Language.RUBY,
            '.swift': Language.SWIFT,
            '.proto': Language.PROTO,
            
            # SQL/Solidity
            '.sql': Language.SOL,
            '.sol': Language.SOL,
            
            # Markup/Documentation
            '.md': Language.MARKDOWN,
            '.markdown': Language.MARKDOWN,
            '.rst': Language.RST,
            '.html': Language.HTML,
            '.htm': Language.HTML,
            '.tex': Language.LATEX,
            '.latex': Language.LATEX,
            
            # Functional Languages
            '.hs': Language.HASKELL,
            '.lhs': Language.HASKELL,
            '.ex': Language.ELIXIR,
            '.exs': Language.ELIXIR,
            '.lua': Language.LUA,
            '.pl': Language.PERL,
            '.pm': Language.PERL,
            
            # Legacy/Other
            '.cob': Language.COBOL,
            '.cbl': Language.COBOL,
            '.ps1': Language.POWERSHELL,
            '.psm1': Language.POWERSHELL,
        }
    
    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunking especializado para código usando RecursiveCharacterTextSplitter."""
        filename = metadata.get('filename', '')
        extension = metadata.get('extension', '')
        
        # Detectar linguagem baseada na extensão
        language = self.language_map.get(extension.lower())
        
        if language:
            # Usar splitter específico da linguagem
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            # Fallback para splitter genérico
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
        
        # Adicionar contexto do arquivo ao conteúdo
        enhanced_content = f"# Arquivo: {filename}\n# Linguagem: {extension}\n\n{content}"
        
        chunks = splitter.split_text(enhanced_content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_type': 'code',
                'language_detected': language.value if language else 'unknown',
                'has_syntax_context': bool(language)
            })
            
            documents.append({
                'text': chunk,
                'metadata': chunk_metadata
            })
        
        return documents


class DocumentationChunkingStrategy(ChunkingStrategy):
    """Estratégia de chunking para documentação e markdown."""
    
    def __init__(self, chunk_size: int = 3000, chunk_overlap: int = 300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunking semântico para documentação."""
        # Usar splitter específico para markdown
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        chunks = splitter.split_text(content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            
            # Extrair cabeçalhos do chunk para contexto
            headers = self._extract_headers(chunk)
            
            chunk_metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_type': 'documentation',
                'headers': headers,
                'has_code_blocks': '```' in chunk,
                'has_links': '[' in chunk and '](' in chunk
            })
            
            documents.append({
                'text': chunk,
                'metadata': chunk_metadata
            })
        
        return documents
    
    def _extract_headers(self, text: str) -> List[str]:
        """Extrai cabeçalhos markdown do texto."""
        header_pattern = r'^(#{1,6})\s+(.+)$'
        headers = []
        
        for line in text.split('\n'):
            match = re.match(header_pattern, line.strip())
            if match:
                level = len(match.group(1))
                title = match.group(2)
                headers.append(f"H{level}: {title}")
        
        return headers


class IssueChunkingStrategy(ChunkingStrategy):
    """Estratégia de chunking para issues e pull requests."""
    
    def __init__(self, chunk_size: int = 2500, chunk_overlap: int = 250):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunking contextual para issues/PRs."""
        # Dividir em seções lógicas
        sections = self._split_into_sections(content)
        
        documents = []
        section_index = 0
        
        for section_name, section_content in sections:
            if not section_content.strip():
                continue
                
            # Usar splitter com separadores apropriados para issues
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n--- COMENTÁRIOS ---\n", "\n\nCOMENTÁRIO #", "\n\n", "\n", " ", ""]
            )
            
            chunks = splitter.split_text(section_content)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_index': section_index,
                    'total_chunks': len(chunks),
                    'chunk_type': 'issue_discussion',
                    'section': section_name,
                    'section_index': i,
                    'has_code_snippets': '```' in chunk,
                    'has_mentions': '@' in chunk,
                    'is_comment_section': 'COMENTÁRIO #' in chunk
                })
                
                documents.append({
                    'text': chunk,
                    'metadata': chunk_metadata
                })
                section_index += 1
        
        return documents
    
    def _split_into_sections(self, content: str) -> List[Tuple[str, str]]:
        """Divide o conteúdo de issues em seções lógicas."""
        sections = []
        
        # Dividir entre descrição principal e comentários
        if "--- COMENTÁRIOS ---" in content:
            parts = content.split("--- COMENTÁRIOS ---", 1)
            sections.append(("description", parts[0]))
            if len(parts) > 1:
                sections.append(("comments", "--- COMENTÁRIOS ---" + parts[1]))
        else:
            sections.append(("description", content))
        
        return sections


class SmartChunker:
    """Chunker inteligente que seleciona a estratégia apropriada baseada no tipo de conteúdo."""
    
    def __init__(self, 
                 code_chunk_size: int = 2000,
                 doc_chunk_size: int = 3000, 
                 issue_chunk_size: int = 2500):
        self.strategies = {
            'code': CodeChunkingStrategy(chunk_size=code_chunk_size),
            'documentation': DocumentationChunkingStrategy(chunk_size=doc_chunk_size),
            'issue': IssueChunkingStrategy(chunk_size=issue_chunk_size),
            'pull_request': IssueChunkingStrategy(chunk_size=issue_chunk_size)
        }
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processa uma lista de documentos aplicando a estratégia apropriada para cada um."""
        chunked_documents = []
        
        for doc in documents:
            content = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            # Determinar estratégia baseada nos metadados
            strategy = self._select_strategy(metadata)
            
            # Aplicar chunking
            chunks = strategy.chunk(content, metadata)
            chunked_documents.extend(chunks)
        
        return chunked_documents
    
    def _select_strategy(self, metadata: Dict[str, Any]) -> ChunkingStrategy:
        """Seleciona a estratégia de chunking apropriada baseada nos metadados."""
        source = metadata.get('source', '')
        filename = metadata.get('filename', '')
        extension = metadata.get('extension', '')
        
        # Priorizar por tipo de fonte
        if source in ['issue', 'pull_request']:
            return self.strategies[source]
        elif source == 'code' or extension in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.rb']:
            return self.strategies['code']
        elif extension in ['.md', '.rst', '.txt'] or 'readme' in filename.lower():
            return self.strategies['documentation']
        else:
            # Fallback para documentação
            return self.strategies['documentation']
    
    def get_chunk_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Retorna estatísticas sobre os chunks gerados."""
        if not documents:
            return {}
        
        total_chunks = len(documents)
        chunk_types = {}
        languages = {}
        avg_chunk_size = 0
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            chunk_type = metadata.get('chunk_type', 'unknown')
            language = metadata.get('language_detected', 'unknown')
            
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            languages[language] = languages.get(language, 0) + 1
            avg_chunk_size += len(doc.get('text', ''))
        
        avg_chunk_size = avg_chunk_size / total_chunks if total_chunks > 0 else 0
        
        return {
            'total_chunks': total_chunks,
            'chunk_types': chunk_types,
            'languages_detected': languages,
            'average_chunk_size': round(avg_chunk_size),
            'chunking_strategies_used': len(set(chunk_types.keys()))
        }