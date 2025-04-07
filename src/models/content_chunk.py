from dataclasses import dataclass
from typing import Dict, List, Any, Optional


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
        return f"ContentChunk(path={'.'.join(self.header_path)}, type={self.chunk_type}, content={self.content[:30]}...)"
