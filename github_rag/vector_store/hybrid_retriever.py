"""
Sistema de recuperação híbrida combinando embeddings densos (semânticos) 
com busca esparsa (BM25) para melhor cobertura e precisão.
"""
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict
import re

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None

from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever


@dataclass
class HybridSearchResult:
    """Resultado da busca híbrida com scores detalhados."""
    document: Document
    dense_score: float
    sparse_score: float
    combined_score: float
    rank_dense: int
    rank_sparse: int
    rank_combined: int


class BM25Retriever:
    """Implementa busca BM25 para retrieval esparso."""
    
    def __init__(self, documents: List[Document], k1: float = 1.2, b: float = 0.75):
        """
        Inicializa o retriever BM25.
        
        Args:
            documents: Lista de documentos para indexação
            k1: Parâmetro de saturação de termo (padrão: 1.2)
            b: Parâmetro de normalização de comprimento (padrão: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.bm25 = None
        self.tokenized_docs = None
        self.logger = logging.getLogger(__name__)
        
        if not BM25_AVAILABLE:
            self.logger.warning(
                "rank-bm25 não está disponível. "
                "Busca esparsa será desabilitada. "
                "Execute: pip install rank-bm25"
            )
        else:
            self._build_index()
    
    def _build_index(self):
        """Constrói o índice BM25."""
        if not BM25_AVAILABLE or not self.documents:
            return
        
        try:
            # Tokenizar documentos
            self.logger.info(f"Construindo índice BM25 para {len(self.documents)} documentos...")
            self.tokenized_docs = [self._tokenize(doc.page_content) for doc in self.documents]
            
            # Criar índice BM25
            self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)
            self.logger.info("Índice BM25 construído com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao construir índice BM25: {e}")
            self.bm25 = None
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokeniza texto para BM25."""
        # Normalizar e tokenizar
        text = text.lower()
        
        # Manter tokens de código (preservar underscores e pontos)
        # Dividir em palavras, mantendo alguns caracteres especiais importantes para código
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[a-zA-Z]+|\d+', text)
        
        # Filtrar tokens muito curtos (menos de 2 caracteres) exceto dígitos
        tokens = [token for token in tokens if len(token) >= 2 or token.isdigit()]
        
        return tokens
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        Busca documentos usando BM25.
        
        Args:
            query: Query de busca
            top_k: Número de documentos a retornar
        
        Returns:
            Lista de (documento, score) ordenada por relevância
        """
        if not self.bm25:
            return []
        
        try:
            # Tokenizar query
            query_tokens = self._tokenize(query)
            
            if not query_tokens:
                return []
            
            # Calcular scores BM25
            scores = self.bm25.get_scores(query_tokens)
            
            # Criar lista de (índice, score) e ordenar
            scored_docs = [(i, score) for i, score in enumerate(scores)]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Retornar top_k documentos com scores
            results = []
            for i, (doc_idx, score) in enumerate(scored_docs[:top_k]):
                if score > 0:  # Apenas documentos com score positivo
                    results.append((self.documents[doc_idx], float(score)))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erro na busca BM25: {e}")
            return []
    
    def is_available(self) -> bool:
        """Verifica se BM25 está disponível."""
        return self.bm25 is not None


class HybridRetriever(BaseRetriever):
    """Retriever híbrido que combina busca densa (embeddings) e esparsa (BM25)."""
    
    def __init__(self,
                 dense_retriever: BaseRetriever,
                 documents: List[Document],
                 dense_weight: float = 0.7,
                 sparse_weight: float = 0.3,
                 rrf_k: int = 60,
                 top_k_dense: int = 20,
                 top_k_sparse: int = 20,
                 final_top_k: int = 10):
        """
        Inicializa o retriever híbrido.
        
        Args:
            dense_retriever: Retriever baseado em embeddings
            documents: Lista de documentos para BM25
            dense_weight: Peso da busca densa na combinação final
            sparse_weight: Peso da busca esparsa na combinação final
            rrf_k: Parâmetro k para Reciprocal Rank Fusion
            top_k_dense: Documentos a recuperar da busca densa
            top_k_sparse: Documentos a recuperar da busca esparsa
            final_top_k: Número final de documentos a retornar
        """
        self.dense_retriever = dense_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.final_top_k = final_top_k
        self.logger = logging.getLogger(__name__)
        
        # Inicializar BM25 retriever
        self.bm25_retriever = BM25Retriever(documents)
        
        # Validar pesos
        if abs(dense_weight + sparse_weight - 1.0) > 1e-6:
            self.logger.warning(f"Pesos não somam 1.0: dense={dense_weight}, sparse={sparse_weight}")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Implementa busca híbrida combinando dense e sparse retrieval."""
        return self.hybrid_search(query, top_k=self.final_top_k)
    
    def hybrid_search(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Executa busca híbrida combinando resultados densos e esparsos.
        
        Args:
            query: Query de busca
            top_k: Número de documentos a retornar (usa self.final_top_k se None)
        
        Returns:
            Lista de documentos rankeados pela combinação híbrida
        """
        if top_k is None:
            top_k = self.final_top_k
        
        # Busca densa (embeddings)
        try:
            dense_docs = self.dense_retriever.get_relevant_documents(query)[:self.top_k_dense]
            self.logger.debug(f"Busca densa retornou {len(dense_docs)} documentos")
        except Exception as e:
            self.logger.error(f"Erro na busca densa: {e}")
            dense_docs = []
        
        # Busca esparsa (BM25)
        sparse_results = []
        if self.bm25_retriever.is_available():
            try:
                sparse_results = self.bm25_retriever.search(query, self.top_k_sparse)
                self.logger.debug(f"Busca esparsa retornou {len(sparse_results)} documentos")
            except Exception as e:
                self.logger.error(f"Erro na busca esparsa: {e}")
        
        # Se apenas uma busca funcionou, usar ela
        if not dense_docs and not sparse_results:
            self.logger.warning("Ambas as buscas falharam")
            return []
        elif not sparse_results:
            self.logger.info("Usando apenas busca densa")
            return dense_docs[:top_k]
        elif not dense_docs:
            self.logger.info("Usando apenas busca esparsa")
            return [doc for doc, _ in sparse_results[:top_k]]
        
        # Combinar resultados usando RRF (Reciprocal Rank Fusion)
        combined_results = self._combine_with_rrf(dense_docs, sparse_results, top_k)
        
        return [result.document for result in combined_results]
    
    def hybrid_search_with_scores(self, query: str, top_k: Optional[int] = None) -> List[HybridSearchResult]:
        """
        Busca híbrida retornando scores detalhados.
        
        Args:
            query: Query de busca
            top_k: Número de documentos a retornar
            
        Returns:
            Lista de HybridSearchResult com scores detalhados
        """
        if top_k is None:
            top_k = self.final_top_k
        
        # Executar buscas
        dense_docs = self.dense_retriever.get_relevant_documents(query)[:self.top_k_dense]
        sparse_results = self.bm25_retriever.search(query, self.top_k_sparse) if self.bm25_retriever.is_available() else []
        
        # Combinar com scores detalhados
        return self._combine_with_rrf(dense_docs, sparse_results, top_k)
    
    def _combine_with_rrf(self, 
                         dense_docs: List[Document], 
                         sparse_results: List[Tuple[Document, float]], 
                         top_k: int) -> List[HybridSearchResult]:
        """
        Combina resultados usando Reciprocal Rank Fusion.
        
        RRF Score = 1/(k + rank_dense) + 1/(k + rank_sparse)
        onde k é um parâmetro de suavização (tipicamente 60).
        """
        # Criar mapas de documento para rank e score
        doc_to_dense_rank = {self._doc_id(doc): i + 1 for i, doc in enumerate(dense_docs)}
        doc_to_sparse_rank = {self._doc_id(doc): i + 1 for i, (doc, _) in enumerate(sparse_results)}
        
        sparse_scores = {self._doc_id(doc): score for doc, score in sparse_results}
        
        # Coletar todos os documentos únicos
        all_docs = {}
        for doc in dense_docs:
            doc_id = self._doc_id(doc)
            all_docs[doc_id] = doc
        
        for doc, _ in sparse_results:
            doc_id = self._doc_id(doc)
            all_docs[doc_id] = doc
        
        # Calcular scores RRF combinados
        rrf_results = []
        for doc_id, doc in all_docs.items():
            dense_rank = doc_to_dense_rank.get(doc_id, float('inf'))
            sparse_rank = doc_to_sparse_rank.get(doc_id, float('inf'))
            
            # Calcular RRF score
            rrf_score = 0.0
            if dense_rank != float('inf'):
                rrf_score += 1.0 / (self.rrf_k + dense_rank)
            if sparse_rank != float('inf'):
                rrf_score += 1.0 / (self.rrf_k + sparse_rank)
            
            # Scores individuais (0 se documento não apareceu na busca)
            dense_score = 1.0 / dense_rank if dense_rank != float('inf') else 0.0
            sparse_score = sparse_scores.get(doc_id, 0.0)
            
            rrf_results.append(HybridSearchResult(
                document=doc,
                dense_score=dense_score,
                sparse_score=sparse_score,
                combined_score=rrf_score,
                rank_dense=dense_rank if dense_rank != float('inf') else -1,
                rank_sparse=sparse_rank if sparse_rank != float('inf') else -1,
                rank_combined=0  # Será preenchido após ordenação
            ))
        
        # Ordenar por score RRF
        rrf_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Atualizar ranks combinados
        for i, result in enumerate(rrf_results):
            result.rank_combined = i + 1
        
        self.logger.debug(
            f"RRF combinou {len(dense_docs)} docs densos e {len(sparse_results)} "
            f"esparsos em {len(rrf_results)} únicos"
        )
        
        return rrf_results[:top_k]
    
    def _doc_id(self, doc: Document) -> str:
        """Gera ID único para um documento baseado no conteúdo."""
        # Usar hash do conteúdo como ID único
        content = doc.page_content[:200]  # Primeiros 200 caracteres
        return str(hash(content))
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas dos retrievers."""
        return {
            "dense_retriever": {
                "type": type(self.dense_retriever).__name__,
                "available": True
            },
            "sparse_retriever": {
                "type": "BM25Okapi",
                "available": self.bm25_retriever.is_available(),
                "num_documents": len(self.bm25_retriever.documents) if self.bm25_retriever.is_available() else 0
            },
            "hybrid_config": {
                "dense_weight": self.dense_weight,
                "sparse_weight": self.sparse_weight,
                "rrf_k": self.rrf_k,
                "top_k_dense": self.top_k_dense,
                "top_k_sparse": self.top_k_sparse,
                "final_top_k": self.final_top_k
            }
        }