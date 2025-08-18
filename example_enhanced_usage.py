"""
Exemplo de uso do sistema RAG aprimorado com todas as melhorias implementadas.
Demonstra como usar chunking inteligente, retrieval híbrido, reranking e query routing.
"""
import os
from dotenv import load_dotenv
from github_rag.vector_store.enhanced_vector_store import EnhancedVectorStore
from github_rag.data_loaders.data_loader import GitHubDataLoader
from github_rag.clients.github_client import GitHubClient
from github_rag.managers.conversation import ConversationManager
from github_rag.vector_store.vector_retrieval import VectorRetrieval

def main():
    """Exemplo de uso completo do sistema RAG aprimorado."""
    
    # Carregar variáveis de ambiente
    load_dotenv()
    
    # Configuração
    repo_url = "https://github.com/langchain-ai/langchain"  # Exemplo
    
    print("🚀 Inicializando sistema RAG aprimorado...")
    
    # 1. Criar cliente GitHub
    github_client = GitHubClient(
        repo_url=repo_url,
        token=os.environ.get("GITHUB_API_TOKEN")
    )
    
    # 2. Inicializar data loader com chunking inteligente
    print("📥 Configurando carregamento de dados com chunking inteligente...")
    data_loader = GitHubDataLoader(
        github_client=github_client,
        use_smart_chunking=True  # Habilitar chunking inteligente
    )
    
    # 3. Carregar dados
    print("🔍 Carregando dados do repositório...")
    data_loader.load_data(
        content_types=["code", "issue"],
        limit_issues=50,  # Limitar para exemplo
        max_files=20
    )
    
    # 4. Criar chunks inteligentes
    print("🧠 Processando documentos com chunking inteligente...")
    documents = data_loader.create_text_chunks(
        chunk_size=2000,
        chunk_overlap=200
    )
    
    print(f"✅ Processados {len(documents)} chunks inteligentes")
    
    # 5. Inicializar vector store aprimorado
    print("🔧 Inicializando vector store aprimorado...")
    vector_store = EnhancedVectorStore(
        embeddings_model=os.environ.get("OPENAI_EMBBENDING_MODEL"),
        persist_directory="./enhanced_rag_db",
        collection_name=f"enhanced_{repo_url.split('/')[-1]}",
        enable_hybrid_search=True,      # Busca híbrida (denso + BM25)
        enable_reranking=True,          # Reranking com cross-encoder
        enable_smart_routing=True       # Query routing inteligente
    )
    
    # 6. Criar base de dados vetorial
    print("🗄️ Criando base de dados vetorial...")
    success = vector_store.create_vector_db(
        documents=documents,
        batch_size=50,
        show_progress=True
    )
    
    if not success:
        print("❌ Falha ao criar base vetorial")
        return
    
    # 7. Verificar estatísticas do sistema
    stats = vector_store.get_system_stats()
    print("📊 Estatísticas do sistema:")
    print(f"   - Documentos indexados: {stats.get('total_documentos', 0)}")
    print(f"   - Busca híbrida: {'✅' if stats['enhancements']['hybrid_search']['available'] else '❌'}")
    print(f"   - Reranking: {'✅' if stats['enhancements']['reranking']['available'] else '❌'}")
    print(f"   - Query routing: {'✅' if stats['enhancements']['smart_routing']['available'] else '❌'}")
    
    # 8. Criar retriever aprimorado
    enhanced_retriever = vector_store.get_enhanced_retriever()
    
    # 9. Inicializar conversation manager com recursos avançados
    print("💬 Configurando gerenciador de conversação...")
    conversation_manager = ConversationManager(
        retriever=enhanced_retriever,
        model_name=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        memory_enabled=True,
        memory_window=5,
        enable_smart_prompts=True,      # Templates especializados
        enable_query_routing=True,      # Routing inteligente
        verbose=True
    )
    
    print("\n" + "="*60)
    print("🎯 Sistema RAG Aprimorado Pronto!")
    print("="*60)
    
    # 10. Demonstrar diferentes tipos de consultas
    demo_queries = [
        # Query de explicação de código
        {
            "query": "Explain how the LangChain document loaders work",
            "expected_type": "code_explanation"
        },
        
        # Query de implementação
        {
            "query": "How to implement a custom retriever in LangChain?",
            "expected_type": "code_implementation"
        },
        
        # Query de debugging
        {
            "query": "Error when loading documents - import module not found",
            "expected_type": "debugging"
        },
        
        # Query de documentação
        {
            "query": "What are the available text splitters in LangChain?",
            "expected_type": "documentation"
        }
    ]
    
    for i, demo in enumerate(demo_queries, 1):
        print(f"\n🔍 Demo Query {i}: {demo['query']}")
        print(f"📝 Tipo esperado: {demo['expected_type']}")
        print("-" * 50)
        
        # Usar enhanced_query para demonstrar todos os recursos
        if hasattr(vector_store, 'enhanced_query'):
            # Query com todas as melhorias
            enhanced_result = vector_store.enhanced_query(
                query_text=demo['query'],
                limit=5
            )
            
            print(f"🧠 Método usado: {enhanced_result.get('method_used', 'N/A')}")
            
            if enhanced_result.get('query_classification'):
                class_info = enhanced_result['query_classification']
                print(f"🎯 Tipo detectado: {class_info['type']} (confiança: {class_info['confidence']:.2f})")
                print(f"🔑 Keywords: {', '.join(class_info.get('keywords', [])[:3])}")
            
            print(f"📚 Documentos encontrados: {len(enhanced_result.get('documents', []))}")
            
            if enhanced_result.get('rerank_stats', {}).get('reranker_available'):
                rerank_stats = enhanced_result['rerank_stats']
                print(f"🔄 Reranking aplicado: {rerank_stats.get('documents_reranked', 0)} docs")
        else:
            # Fallback para query tradicional
            result = conversation_manager.query(demo['query'])
            print(f"💬 Resposta: {result.get('resposta', '')[:100]}...")
            print(f"📋 Fontes: {len(result.get('fontes', []))}")
        
        print()
    
    print("\n" + "="*60)
    print("✨ Demonstração concluída!")
    print("="*60)
    
    # Mostrar estatísticas finais
    final_stats = conversation_manager.get_stats()
    print(f"📈 Consultas processadas: {final_stats.get('queries', 0)}")
    print(f"⚡ Features inteligentes ativas:")
    smart_features = final_stats.get('smart_features', {})
    for feature, enabled in smart_features.items():
        if feature.endswith('_available'):
            continue
        status = "✅" if enabled else "❌"
        print(f"   - {feature.replace('_', ' ').title()}: {status}")


if __name__ == "__main__":
    main()