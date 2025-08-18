import os
import json
import time
import argparse
from typing import Dict, Any
from dotenv import load_dotenv
from github_rag import SessionManager
from github_rag.utils import setup_environment


def main():
    """Função principal para analisar qualquer repositório do GitHub usando RAG"""
    parser = argparse.ArgumentParser(description="GitHub RAG Tool")
    parser.add_argument(
        "--repo_url", type=str, help="URL do repositório GitHub para análise"
    )
    args = parser.parse_args()

    # Carregar variáveis de ambiente
    if not setup_environment():
        print("❌ Falha ao configurar o ambiente")
        return

    # Obter URL do repositório (da linha de comando ou input do usuário)
    repo_url = args.repo_url
    if not repo_url:
        repo_url = input("Digite a URL do repositório GitHub: ").strip()

    if not repo_url.startswith("https://github.com/"):
        print("❌ URL inválida. Use o formato: https://github.com/username/repo")
        return

    print(f"🚀 Iniciando sessão com o repositório: {repo_url}")

    # Extrair nome do repositório para uso em mensagens e nome de sessão
    repo_name = repo_url.split("/")[-1]

    # Configurações pré-definidas
    config_options = {
        "chunk_size": 50_000,  # Chunks um pouco maiores para capturar mais contexto
        "chunk_overlap": 3_000,  # Sobreposição maior para evitar perda de informação
        "retriever_k": 30,  # Mais documentos para uma cobertura mais ampla
        "use_memory": True,  # Habilitar memória da conversa
        "memory_window": 5,  # Janela de memória moderada
    }

    # Criar a ferramenta RAG
    print("🔧 Configurando sessão RAG...")
    start_time = time.time()
    session_manager = SessionManager(
        repo_url=repo_url,
        initial_config=config_options,
        embeddings_model=os.environ.get("OPENAI_EMBBENDING_MODEL"),
    )

    # Aplicar configurações
    # Configurações aplicadas via SessionManager
    print(f"⚙️ Configurações aplicadas: {json.dumps(config_options, indent=2)}")

    # Sempre reconstruir a base de conhecimento
    rebuild = True

    # Construir base de conhecimento
    print("🔍 Construindo a sessão RAG...")
    success = session_manager.setup(
        limit_issues=100, rebuild=rebuild  # Limitamos a 100 issues
    )

    if not success:
        print("❌ Falha ao construir a base de conhecimento")
        return

    setup_time = time.time() - start_time
    print(f"✅ Preparação concluída em {setup_time:.2f} segundos")

    # Mostrar status da ferramenta
    status = session_manager.get_status()
    print("\n📊 Status da Ferramenta:")
    print(f"- Sessão: {status['session_id']}")
    print(f"- Modelo de Chat: {os.environ.get('OPENAI_MODEL')}")
    print(f"- Modelo de Embedding: {os.environ.get('OPENAI_EMBBENDING_MODEL')}")
    print(f"- Base vetorial pronta: {status['is_vectordb_ready']}")

    if status["vector_db"]:
        print(f"- Documentos indexados: {status['vector_db']['total_documentos']}")

    # Loop de consulta
    print(f"\n💬 Modo de consulta ativado para o repositório {repo_name}")
    print(
        "Digite 'sair' para encerrar, 'status' para ver estatísticas, ou 'ajuda' para comandos adicionais"
    )

    while True:
        question = input("\nPergunta: ")

        # Comandos especiais
        if question.lower() in ["sair", "exit", "quit"]:
            break
        elif question.lower() == "status":
            current_status = session_manager.get_status()
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
            sources = session_manager.search_sources(query, limit=10)
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
        result = session_manager.query(question)
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
                    f"  [{i}] {fonte.get('tipo', 'Tipo desconhecido')}: {fonte.get('título', 'Sem título')}"
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
    save_dir = f"./sessions/{repo_name}_{int(time.time())}"
    print(f"\n💾 Salvando sessão em {save_dir}...")
    success = session_manager.save_session(save_dir)

    if success:
        print("✅ Sessão salva com sucesso")
    else:
        print("⚠️ Falha ao salvar a sessão")

    print("\n🎬 Sessão finalizada. Obrigado por utilizar a ferramenta!")


if __name__ == "__main__":
    main()
