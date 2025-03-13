import os
from dotenv import load_dotenv
from github_rag import GitHubRagTool
from github_rag.utils import setup_environment


def main():
    """Função principal de exemplo"""
    # Carregar variáveis de ambiente
    if not setup_environment():
        print("❌ Falha ao configurar o ambiente")
        return

    # Obter URL do repositório
    repo_url = input("Digite a URL do repositório GitHub: ")

    # Criar a ferramenta RAG
    rag_tool = GitHubRagTool(
        repo_url=repo_url, content_types=["code", "issue"], custom_model="gpt-3.5-turbo"
    )

    # Construir base de conhecimento sem limites
    rag_tool.build_knowledge_base()

    # Loop de consulta
    print(
        "\n💬 Agora você pode fazer perguntas sobre o repositório. Digite 'sair' para encerrar."
    )
    while True:
        question = input("\nPergunta: ")
        if question.lower() in ["sair", "exit", "quit"]:
            break

        result = rag_tool.query(question)

        print("\nResposta:")
        print(result["resposta"])

        if result["fontes"]:
            print("\nFontes:")
            for fonte in result["fontes"]:
                if fonte["tipo"] == "Issue":
                    print(f"- Issue #{fonte['número']}: {fonte['url']}")
                else:
                    print(f"- Arquivo: {fonte['arquivo']}")


if __name__ == "__main__":
    main()
