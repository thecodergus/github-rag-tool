from src.data_extraction.github_extractor import GitHubDataExtractor
from src.document_processing.chunker import HierarchicalChunker
from src.models.content_chunk import ContentChunk
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    # Uso do código
    extractor = GitHubDataExtractor(
        github_api_token=os.environ.get("GITHUB_API_TOKEN"),
        repo_url="https://github.com/NVIDIA/Isaac-GR00T",
    )
    chunker = HierarchicalChunker(config={"max_chunk_size": 3100})

    # 1. Procurar os dados
    dados = extractor.get_structured_data(["issues", "documentation"])

    # 2. Procuessar os dados para criar uma RAG hierarquica
    print(dados)
