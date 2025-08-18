import os
from typing import List, Dict, Any

class CodeProcessor:
    """
    Processa arquivos de código para criar documentos de texto.
    """

    def __init__(self, text_splitter):
        self.text_splitter = text_splitter

    def process(self, code_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processa a lista de arquivos de código em documentos textuais.
        """
        if not code_files:
            return []
        documents: List[Dict[str, Any]] = []
        for file_info in code_files:
            content = file_info.get("content", "")
            if not content:
                continue
            # Adiciona contexto de nome de arquivo
            enhanced_content = f"ARQUIVO: {file_info.get('name')}\n\n{content}"
            # Divide o texto em chunks
            chunks = self.text_splitter.split_text(enhanced_content)
            # Metadados iniciais
            metadata: Dict[str, Any] = {
                "source": "code",
                "filename": file_info.get("name"),
                "url": file_info.get("url"),
                "extension": os.path.splitext(file_info.get("name", ""))[1],
            }
            # Criar documentos para cada chunk
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                documents.append({"text": chunk, "metadata": chunk_metadata})
        return documents