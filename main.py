import os
import json
import time
from typing import Dict, Any
from dotenv import load_dotenv
from github_rag import GitHubRagTool
from github_rag.utils import setup_environment


def test_lerobot():
    """Função de teste específica para o repositório lerobot da Hugging Face"""
    # Carregar variáveis de ambiente
    if not setup_environment():
        print("❌ Falha ao configurar o ambiente")
        return

    # Configuração estática
    repo_url = "https://github.com/huggingface/lerobot"

    print(f"🚀 Iniciando sessão de teste com o repositório: {repo_url}")

    # Configurações pré-definidas para teste
    config_options = {
        "chunk_size": 1200,  # Chunks um pouco maiores para capturar mais contexto
        "chunk_overlap": 300,  # Sobreposição maior para evitar perda de informação
        "retriever_k": 7,  # Mais documentos para uma cobertura mais ampla
        "use_memory": True,  # Habilitar memória da conversa
        "memory_window": 5,  # Janela de memória moderada
    }

    # Criar a ferramenta RAG
    print("🔧 Inicializando a ferramenta RAG...")
    start_time = time.time()
    rag_tool = GitHubRagTool(
        repo_url=repo_url,
        content_types=[
            "code",
            "issue",
            "pull_request",
        ],  # Incluindo PRs para mais contexto
        custom_model=os.environ.get("OPENAI_EMBBENDING_MODEL"),
        temperature=0.2,  # Temperatura mais baixa para respostas mais consistentes
    )

    # Aplicar configurações de teste
    rag_tool.configure(config_options)
    print(f"⚙️ Configurações aplicadas: {json.dumps(config_options, indent=2)}")

    # Verificar se devemos reconstruir a base
    rebuild = (
        input("Reconstruir a base de conhecimento? (s/n, padrão: n): ").strip().lower()
        == "s"
    )

    # Construir base de conhecimento
    print("🔍 Construindo base de conhecimento...")
    success = rag_tool.build_knowledge_base(
        limit_issues=100, rebuild=rebuild  # Para teste, limitamos a 100 issues
    )

    if not success:
        print("❌ Falha ao construir a base de conhecimento")
        return

    setup_time = time.time() - start_time
    print(f"✅ Preparação concluída em {setup_time:.2f} segundos")

    # Mostrar status da ferramenta
    status = rag_tool.get_status()
    print("\n📊 Status da Ferramenta:")
    print(f"- Sessão: {status['session_id']}")
    print(f"- Modelo de Chat: {os.environ.get("OPENAI_MODEL")}")
    print(f"- Modelo de Embedding: {os.environ.get("OPENAI_EMBBENDING_MODEL")}")
    print(f"- Base vetorial pronta: {status['is_vectordb_ready']}")

    if status["vector_db"]:
        print(f"- Documentos indexados: {status['vector_db']["total_documentos"]}")

    # Loop de consulta
    print("\n💬 Modo de teste ativado para o repositório lerobot")
    print(
        "Digite 'sair' para encerrar, 'status' para ver estatísticas, ou 'ajuda' para comandos adicionais"
    )

    while True:
        question = input("\nPergunta: ")

        # Comandos especiais
        if question.lower() in ["sair", "exit", "quit"]:
            break
        elif question.lower() == "status":
            current_status = rag_tool.get_status()
            print("\n📊 Estatísticas Atuais:")
            print(f"- Consultas realizadas: {current_status['stats']['queries_count']}")
            print(
                f"- Tempo médio de resposta: {current_status['stats']['avg_response_time']:.2f}s"
            )
            continue
        elif question.lower() == "ajuda":
            print("\n📋 Comandos disponíveis:")
            print("- 'sair': Encerra o programa")
            print("- 'status': Mostra estatísticas atuais")
            print("- 'fontes <consulta>': Busca fontes diretamente sem gerar resposta")
            print("- 'ajuda': Mostra esta mensagem")
            continue
        elif question.lower().startswith("fontes "):
            query = question[7:].strip()  # Remove o comando "fontes "
            sources = rag_tool.search_sources(query, limit=10)
            print("\n📚 Fontes encontradas:")
            for source in sources:
                print(f"- [{source['index']}] Score: {source['score']:.4f}")
                print(f"  Tipo: {source['metadata'].get('type', 'N/A')}")
                if source["metadata"].get("file_path"):
                    print(f"  Arquivo: {source['metadata']['file_path']}")
                if source["metadata"].get("number"):
                    print(f"  Número: {source['metadata']['number']}")
                print()
            continue

        # Consulta normal
        start_query_time = time.time()
        print("⏳ Processando consulta...")
        result = rag_tool.query(question)
        query_time = time.time() - start_query_time

        # Exibir resultado
        print(f"\n🔄 Resposta (gerada em {query_time:.2f}s):")
        print(result.get("resposta", "Resposta não fornecida"))

        # Imprimir as fontes do conhecimento
        fontes = result.get("fontes", [])
        if fontes:
            print("\n📚 Fontes do conhecimento:")
            for i, fonte in enumerate(fontes, 1):
                print(
                    f"  {i}. {fonte.get('tipo', 'Tipo desconhecido')}: {fonte.get('título', 'Sem título')}"
                )
                if fonte.get("url"):
                    print(f"     URL: {fonte.get('url')}")
                if fonte.get("conteúdo_parcial"):
                    conteudo = fonte.get("conteúdo_parcial")
                    # Limitar o tamanho do conteúdo para melhor visualização
                    if len(conteudo) > 100:
                        conteudo = conteudo[:100] + "..."
                    print(f"     Trecho: {conteudo}")
                print()
        else:
            print("\nNenhuma fonte de conhecimento disponível.")

        print(f"\nConfiança: {result.get('confiança', 'N/A')}")

        # Exibir fontes
        sources = result.get("sources", [])
        if sources:
            print("\n📚 Fontes:")
            for i, source in enumerate(sources, 1):
                metadata = source.get("metadata", {})
                source_type = metadata.get("type", "desconhecido")

                if source_type == "issue":
                    print(
                        f"[{i}] Issue #{metadata.get('number', 'N/A')}: {metadata.get('title', 'Sem título')}"
                    )
                    print(f"    URL: {metadata.get('url', 'N/A')}")
                elif source_type == "pull_request":
                    print(
                        f"[{i}] PR #{metadata.get('number', 'N/A')}: {metadata.get('title', 'Sem título')}"
                    )
                    print(f"    URL: {metadata.get('url', 'N/A')}")
                else:
                    print(f"[{i}] Arquivo: {metadata.get('file_path', 'N/A')}")
                    if "language" in metadata:
                        print(f"    Linguagem: {metadata.get('language', 'N/A')}")

    # Salvar sessão automaticamente
    save_dir = f"./sessions/lerobot_test_{int(time.time())}"
    print(f"\n💾 Salvando sessão em {save_dir}...")
    success = rag_tool.save_session(save_dir)

    if success:
        print("✅ Sessão salva com sucesso")
    else:
        print("⚠️ Falha ao salvar a sessão")

    print("\n🎬 Teste finalizado. Obrigado por utilizar a ferramenta!")


if __name__ == "__main__":
    test_lerobot()
