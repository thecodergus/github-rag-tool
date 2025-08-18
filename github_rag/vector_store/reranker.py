"""
Sistema de reranking para melhorar a relevância dos documentos recuperados.
Utiliza modelos cross-encoder para reordenar resultados baseado na query específica.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    CrossEncoder = None


@dataclass
class RerankingResult:
    """Resultado do processo de reranking."""
    text: str
    metadata: Dict[str, Any]
    original_score: float
    rerank_score: float
    final_score: float
    rank_change: int  # Mudança na posição (positivo = subiu, negativo = desceu)


class DocumentReranker:
    """Reranker de documentos usando modelos cross-encoder."""
    
    def __init__(self, 
                 model_name: str = "ms-marco-MiniLM-L-12-v2",
                 batch_size: int = 32,
                 max_length: int = 512,
                 combine_scores: bool = True,
                 retrieval_weight: float = 0.3,
                 rerank_weight: float = 0.7):
        """
        Inicializa o reranker.
        
        Args:
            model_name: Nome do modelo cross-encoder
            batch_size: Tamanho do batch para processamento
            max_length: Comprimento máximo do texto
            combine_scores: Se deve combinar scores de retrieval e reranking
            retrieval_weight: Peso do score de retrieval na combinação
            rerank_weight: Peso do score de reranking na combinação
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.combine_scores = combine_scores
        self.retrieval_weight = retrieval_weight
        self.rerank_weight = rerank_weight
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo cross-encoder."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning(
                "sentence-transformers não está disponível. "
                "Reranking será desabilitado. "
                "Execute: pip install sentence-transformers"
            )
            return
        
        try:
            # Mapear nomes de modelos para IDs completos
            model_mapping = {
                "ms-marco-MiniLM-L-12-v2": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                "ms-marco-TinyBERT-L-2-v2": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
                "ms-marco-electra-base": "cross-encoder/ms-marco-electra-base"
            }
            
            full_model_name = model_mapping.get(self.model_name, self.model_name)
            
            self.logger.info(f"Carregando modelo de reranking: {full_model_name}")
            self.model = CrossEncoder(full_model_name, max_length=self.max_length)
            self.logger.info("Modelo de reranking carregado com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo de reranking: {e}")
            self.logger.warning("Reranking será desabilitado")
            self.model = None
    
    def rerank(self, 
               query: str, 
               documents: List[Dict[str, Any]], 
               top_k: Optional[int] = None) -> List[RerankingResult]:
        """
        Reordena documentos baseado na relevância para a query.
        
        Args:
            query: Query do usuário
            documents: Lista de documentos com 'text', 'metadata' e opcionalmente 'score'
            top_k: Número de documentos a retornar (None = todos)
        
        Returns:
            Lista de documentos reordenados com scores atualizados
        """
        if not self.model or not documents:
            return self._fallback_ranking(documents, top_k)
        
        try:
            # Preparar pares query-documento
            pairs = []
            original_scores = []
            
            for doc in documents:
                text = doc.get('text', '')[:self.max_length]  # Truncar se necessário
                pairs.append([query, text])
                original_scores.append(doc.get('score', 0.0))
            
            # Calcular scores de reranking
            self.logger.debug(f"Reranking {len(pairs)} documentos...")
            rerank_scores = self.model.predict(pairs, batch_size=self.batch_size)
            
            # Normalizar scores para [0, 1]
            rerank_scores = self._normalize_scores(rerank_scores)
            
            # Combinar scores se habilitado
            if self.combine_scores and any(original_scores):
                final_scores = self._combine_scores(original_scores, rerank_scores)
            else:
                final_scores = rerank_scores
            
            # Criar resultados com informações de ranking
            results = []
            for i, (doc, orig_score, rerank_score, final_score) in enumerate(
                zip(documents, original_scores, rerank_scores, final_scores)
            ):
                results.append(RerankingResult(
                    text=doc.get('text', ''),
                    metadata=doc.get('metadata', {}),
                    original_score=float(orig_score),
                    rerank_score=float(rerank_score),
                    final_score=float(final_score),
                    rank_change=0  # Será calculado após ordenação
                ))
            
            # Ordenar por score final (descendente)
            results.sort(key=lambda x: x.final_score, reverse=True)
            
            # Calcular mudanças de ranking
            for new_rank, result in enumerate(results):
                original_rank = next(
                    i for i, doc in enumerate(documents) 
                    if doc.get('text', '') == result.text
                )
                result.rank_change = original_rank - new_rank
            
            # Aplicar top_k se especificado
            if top_k:
                results = results[:top_k]
            
            self.logger.debug(f"Reranking concluído. Top documento mudou de score {results[0].original_score:.4f} para {results[0].final_score:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erro durante reranking: {e}")
            return self._fallback_ranking(documents, top_k)
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normaliza scores para o range [0, 1]."""
        if not scores:
            return scores
        
        scores_array = np.array(scores)
        
        # Usar sigmoid para normalizar scores do cross-encoder
        normalized = 1 / (1 + np.exp(-scores_array))
        
        return normalized.tolist()
    
    def _combine_scores(self, 
                       retrieval_scores: List[float], 
                       rerank_scores: List[float]) -> List[float]:
        """Combina scores de retrieval e reranking."""
        combined = []
        
        for ret_score, rerank_score in zip(retrieval_scores, rerank_scores):
            # Normalizar score de retrieval se necessário (assumindo que pode ser > 1)
            norm_ret_score = min(ret_score, 1.0)
            
            combined_score = (
                self.retrieval_weight * norm_ret_score + 
                self.rerank_weight * rerank_score
            )
            combined.append(combined_score)
        
        return combined
    
    def _fallback_ranking(self, 
                         documents: List[Dict[str, Any]], 
                         top_k: Optional[int] = None) -> List[RerankingResult]:
        """Fallback quando o modelo não está disponível."""
        results = []
        
        for doc in documents:
            original_score = doc.get('score', 0.0)
            results.append(RerankingResult(
                text=doc.get('text', ''),
                metadata=doc.get('metadata', {}),
                original_score=float(original_score),
                rerank_score=float(original_score),  # Usar score original
                final_score=float(original_score),
                rank_change=0
            ))
        
        # Ordenar por score original
        results.sort(key=lambda x: x.original_score, reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results
    
    def is_available(self) -> bool:
        """Verifica se o reranking está disponível."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo carregado."""
        if not self.model:
            return {"status": "unavailable", "reason": "model_not_loaded"}
        
        return {
            "status": "available",
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "combine_scores": self.combine_scores,
            "weights": {
                "retrieval": self.retrieval_weight,
                "rerank": self.rerank_weight
            }
        }


class AdaptiveReranker(DocumentReranker):
    """Reranker adaptativo que ajusta estratégia baseado no tipo de query."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.query_patterns = {
            'code': [
                'function', 'class', 'method', 'variable', 'import', 'def ', 'async',
                'implementation', 'how to implement', 'code example', 'syntax'
            ],
            'debug': [
                'error', 'bug', 'exception', 'traceback', 'debugging', 'fix',
                'problem', 'issue', 'not working', 'fails'
            ],
            'documentation': [
                'how to', 'what is', 'explain', 'documentation', 'guide',
                'tutorial', 'example', 'usage', 'api'
            ]
        }
    
    def rerank(self, 
               query: str, 
               documents: List[Dict[str, Any]], 
               top_k: Optional[int] = None) -> List[RerankingResult]:
        """Reranking adaptativo baseado no tipo de query."""
        # Detectar tipo de query
        query_type = self._detect_query_type(query.lower())
        
        # Filtrar documentos relevantes para o tipo de query se possível
        filtered_docs = self._filter_by_query_type(documents, query_type)
        
        # Usar documentos filtrados se reduziu significativamente o conjunto
        if len(filtered_docs) < len(documents) * 0.8 and filtered_docs:
            documents = filtered_docs
            self.logger.debug(f"Filtrou documentos para query tipo '{query_type}': {len(documents)} documentos")
        
        return super().rerank(query, documents, top_k)
    
    def _detect_query_type(self, query: str) -> str:
        """Detecta o tipo da query baseado em padrões."""
        scores = {query_type: 0 for query_type in self.query_patterns}
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    scores[query_type] += 1
        
        # Retornar tipo com maior score, ou 'general' se empate
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return 'general'
    
    def _filter_by_query_type(self, 
                             documents: List[Dict[str, Any]], 
                             query_type: str) -> List[Dict[str, Any]]:
        """Filtra documentos baseado no tipo de query."""
        if query_type == 'general':
            return documents
        
        filtered = []
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            text = doc.get('text', '').lower()
            
            include = False
            
            if query_type == 'code':
                # Priorizar documentos de código
                if (metadata.get('source') == 'code' or 
                    metadata.get('chunk_type') == 'code' or
                    any(keyword in text for keyword in ['def ', 'function', 'class ', 'import'])):
                    include = True
                    
            elif query_type == 'debug':
                # Priorizar issues e discussões sobre problemas
                if (metadata.get('source') in ['issue', 'pull_request'] or
                    any(keyword in text for keyword in ['error', 'bug', 'exception', 'fix'])):
                    include = True
                    
            elif query_type == 'documentation':
                # Priorizar documentação e READMEs
                if (metadata.get('chunk_type') == 'documentation' or
                    'readme' in metadata.get('filename', '').lower() or
                    metadata.get('extension') in ['.md', '.rst', '.txt']):
                    include = True
            
            if include:
                filtered.append(doc)
        
        return filtered if filtered else documents  # Fallback para todos se filtro muito restritivo