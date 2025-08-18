"""
Vector store aprimorado que integra todas as melhorias:
- Chunking inteligente
- Retrieval híbrido (denso + BM25)  
- Reranking com cross-encoder
- Query routing adaptativo
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever

from .vector_store import VectorStore
from .hybrid_retriever import HybridRetriever
from .reranker import AdaptiveReranker
from ..managers.query_router import QueryRouter, QueryClassification
from ..managers.prompt_templates import PromptTemplateManager


class EnhancedVectorStore(VectorStore):
    """
    Vector store aprimorado com todas as melhorias de qualidade RAG integradas.
    """
    
    def __init__(self, 
                 embeddings_model: Optional[Any] = None,
                 persist_directory: str = "./github_rag_db",
                 collection_name: str = "github_data",
                 enable_hybrid_search: bool = True,
                 enable_reranking: bool = True,
                 enable_smart_routing: bool = True):
        """
        Inicializa o vector store aprimorado.
        
        Args:
            embeddings_model: Modelo de embeddings
            persist_directory: Diretório de persistência  
            collection_name: Nome da coleção
            enable_hybrid_search: Habilitar busca híbrida (denso + esparso)
            enable_reranking: Habilitar reranking com cross-encoder
            enable_smart_routing: Habilitar roteamento inteligente de queries
        """
        super().__init__(embeddings_model, persist_directory, collection_name)
        
        self.enable_hybrid_search = enable_hybrid_search
        self.enable_reranking = enable_reranking
        self.enable_smart_routing = enable_smart_routing
        
        # Componentes avançados
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.reranker: Optional[AdaptiveReranker] = None
        self.query_router: Optional[QueryRouter] = None
        self.prompt_manager: Optional[PromptTemplateManager] = None
        
        self.logger = logging.getLogger(__name__)
        
        # Inicializar componentes se habilitados
        self._initialize_advanced_components()
    
    def _initialize_advanced_components(self):
        """Inicializa componentes avançados se habilitados."""
        try:
            if self.enable_reranking:
                self.reranker = AdaptiveReranker()
                self.logger.info(f"Reranker inicializado: {self.reranker.is_available()}")
            
            if self.enable_smart_routing:
                self.query_router = QueryRouter()
                self.prompt_manager = PromptTemplateManager()
                self.logger.info("Query router e prompt manager inicializados")
                
        except Exception as e:
            self.logger.error(f"Erro ao inicializar componentes avançados: {e}")
    
    def create_vector_db(self, 
                        documents: List[Dict[str, Any]], 
                        batch_size: int = 100, 
                        show_progress: bool = True) -> bool:
        """
        Cria a base de dados vetorial e inicializa componentes híbridos.
        """
        # Criar base vetorial tradicional
        success = super().create_vector_db(documents, batch_size, show_progress)
        
        if success and self.enable_hybrid_search:
            self._setup_hybrid_retrieval()
        
        return success
    
    def load_vector_db(self, persist_directory: Optional[str] = None) -> bool:
        """
        Carrega base vetorial existente e configura componentes híbridos.
        """
        success = super().load_vector_db(persist_directory)
        
        if success and self.enable_hybrid_search:
            self._setup_hybrid_retrieval()
        
        return success
    
    def _setup_hybrid_retrieval(self):
        """Configura o retrieval híbrido após carregar/criar a base vetorial."""
        if not self.vector_db:
            return
        
        try:
            # Obter todos os documentos para BM25
            all_docs = self._get_all_documents()
            
            if not all_docs:
                self.logger.warning("Nenhum documento encontrado para hybrid retrieval")
                return
            
            # Criar retriever denso básico
            base_retriever = self.get_retriever()
            
            # Inicializar hybrid retriever
            self.hybrid_retriever = HybridRetriever(
                dense_retriever=base_retriever,
                documents=all_docs
            )
            
            self.logger.info(f"Hybrid retriever configurado com {len(all_docs)} documentos")
            
        except Exception as e:
            self.logger.error(f"Erro ao configurar hybrid retrieval: {e}")
            self.enable_hybrid_search = False
    
    def _get_all_documents(self) -> List[Document]:
        """Recupera todos os documentos da base vetorial."""
        if not self.vector_db:
            return []
        
        try:
            # Usar uma query ampla para obter todos os documentos
            results = self.vector_db.similarity_search("", k=self.vector_db._collection.count())
            return results
        except Exception as e:
            self.logger.error(f"Erro ao recuperar documentos: {e}")
            return []
    
    def enhanced_query(self,
                      query_text: str,
                      limit: int = 10,
                      use_routing: bool = None,
                      use_reranking: bool = None) -> Dict[str, Any]:
        """
        Executa query aprimorada com todos os componentes integrados.
        
        Args:
            query_text: Texto da consulta
            limit: Número de documentos a retornar
            use_routing: Forçar uso/não uso do routing (None = usar configuração padrão)
            use_reranking: Forçar uso/não uso do reranking (None = usar configuração padrão)
            
        Returns:
            Dicionário com resultados detalhados incluindo:
            - documents: Documentos encontrados
            - query_classification: Classificação da query (se routing habilitado)
            - retrieval_stats: Estatísticas do processo
            - rerank_stats: Estatísticas do reranking (se habilitado)
        """
        # Determinar configurações para esta query
        routing_enabled = use_routing if use_routing is not None else self.enable_smart_routing
        reranking_enabled = use_reranking if use_reranking is not None else self.enable_reranking
        
        result = {
            'query': query_text,
            'documents': [],
            'query_classification': None,
            'retrieval_stats': {},
            'rerank_stats': {},
            'method_used': 'basic'
        }
        
        try:
            # 1. Classificação da query (se habilitado)
            classification = None
            retrieval_params = {}
            
            if routing_enabled and self.query_router:
                classification, retrieval_params = self.query_router.route_query(query_text)
                result['query_classification'] = {
                    'type': classification.query_type.value,
                    'confidence': classification.confidence,
                    'keywords': classification.keywords_matched,
                    'reasoning': classification.reasoning
                }
                self.logger.debug(f"Query classificada como: {classification.query_type.value}")
            
            # 2. Recuperação de documentos
            if self.enable_hybrid_search and self.hybrid_retriever:
                # Usar busca híbrida
                raw_documents = self._hybrid_search(query_text, retrieval_params, limit)
                result['method_used'] = 'hybrid'
                result['retrieval_stats'] = self.hybrid_retriever.get_retriever_stats()
            else:
                # Fallback para busca tradicional
                raw_documents = self._traditional_search(query_text, limit)
                result['method_used'] = 'traditional'
            
            # 3. Reranking (se habilitado)
            if reranking_enabled and self.reranker and raw_documents:
                reranked_results = self.reranker.rerank(query_text, raw_documents, limit)
                
                # Converter resultados do reranking
                final_documents = []
                for rr_result in reranked_results:
                    doc_dict = {
                        'text': rr_result.text,
                        'metadata': rr_result.metadata,
                        'score': rr_result.final_score,
                        'original_score': rr_result.original_score,
                        'rerank_score': rr_result.rerank_score,
                        'rank_change': rr_result.rank_change
                    }
                    final_documents.append(doc_dict)
                
                result['documents'] = final_documents
                result['method_used'] += '_reranked'
                result['rerank_stats'] = {
                    'reranker_available': True,
                    'documents_reranked': len(reranked_results),
                    'avg_rank_change': sum(abs(r.rank_change) for r in reranked_results) / len(reranked_results)
                }
                
                self.logger.debug(f"Reranking aplicado a {len(reranked_results)} documentos")
            else:
                # Sem reranking
                result['documents'] = raw_documents[:limit]
                result['rerank_stats'] = {'reranker_available': False}
            
            self.logger.info(
                f"Enhanced query concluída: {len(result['documents'])} documentos "
                f"usando método {result['method_used']}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro em enhanced_query: {e}")
            # Fallback para query básica
            return self._fallback_query(query_text, limit)
    
    def _hybrid_search(self, 
                      query_text: str, 
                      retrieval_params: Dict[str, Any], 
                      limit: int) -> List[Dict[str, Any]]:
        """Executa busca híbrida com parâmetros otimizados."""
        if not self.hybrid_retriever:
            return []
        
        try:
            # Aplicar parâmetros otimizados se fornecidos
            if retrieval_params:
                # Criar temporary retriever com parâmetros customizados
                temp_retriever = HybridRetriever(
                    dense_retriever=self.hybrid_retriever.dense_retriever,
                    documents=self.hybrid_retriever.bm25_retriever.documents,
                    dense_weight=retrieval_params.get('dense_weight', 0.7),
                    sparse_weight=retrieval_params.get('sparse_weight', 0.3),
                    top_k_dense=retrieval_params.get('top_k_dense', 20),
                    top_k_sparse=retrieval_params.get('top_k_sparse', 20),
                    final_top_k=retrieval_params.get('final_top_k', limit)
                )
                
                # Usar temporary retriever
                hybrid_results = temp_retriever.hybrid_search_with_scores(query_text, limit)
            else:
                # Usar retriever padrão
                hybrid_results = self.hybrid_retriever.hybrid_search_with_scores(query_text, limit)
            
            # Converter para formato compatível
            documents = []
            for hr_result in hybrid_results:
                doc_dict = {
                    'text': hr_result.document.page_content,
                    'metadata': hr_result.document.metadata,
                    'score': hr_result.combined_score,
                    'dense_score': hr_result.dense_score,
                    'sparse_score': hr_result.sparse_score
                }
                documents.append(doc_dict)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Erro na busca híbrida: {e}")
            return []
    
    def _traditional_search(self, query_text: str, limit: int) -> List[Dict[str, Any]]:
        """Executa busca tradicional como fallback."""
        try:
            results = self.query(query_text, limit=limit)
            
            # Converter formato
            documents = []
            for result in results:
                doc_dict = {
                    'text': result.get('text', ''),
                    'metadata': result.get('metadata', {}),
                    'score': result.get('score', 0.0)
                }
                documents.append(doc_dict)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Erro na busca tradicional: {e}")
            return []
    
    def _fallback_query(self, query_text: str, limit: int) -> Dict[str, Any]:
        """Query de fallback em caso de erro."""
        try:
            documents = self._traditional_search(query_text, limit)
            return {
                'query': query_text,
                'documents': documents,
                'method_used': 'fallback',
                'error': 'Enhanced query failed, using basic search'
            }
        except Exception as e:
            return {
                'query': query_text,
                'documents': [],
                'method_used': 'failed',
                'error': str(e)
            }
    
    def get_enhanced_retriever(self, **kwargs) -> BaseRetriever:
        """
        Retorna o retriever mais avançado disponível.
        
        Returns:
            Hybrid retriever se disponível, senão retriever básico
        """
        if self.enable_hybrid_search and self.hybrid_retriever:
            return self.hybrid_retriever
        else:
            return super().get_retriever(**kwargs)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas detalhadas do sistema aprimorado."""
        base_stats = super().get_stats()
        
        enhanced_stats = {
            **base_stats,
            'enhancements': {
                'hybrid_search': {
                    'enabled': self.enable_hybrid_search,
                    'available': self.hybrid_retriever is not None
                },
                'reranking': {
                    'enabled': self.enable_reranking,
                    'available': self.reranker is not None and self.reranker.is_available()
                },
                'smart_routing': {
                    'enabled': self.enable_smart_routing,
                    'available': self.query_router is not None
                }
            }
        }
        
        # Adicionar stats específicos dos componentes
        if self.hybrid_retriever:
            enhanced_stats['hybrid_retriever'] = self.hybrid_retriever.get_retriever_stats()
        
        if self.reranker:
            enhanced_stats['reranker'] = self.reranker.get_model_info()
        
        return enhanced_stats