"""
Sistema de roteamento inteligente de consultas que classifica automaticamente
o tipo de pergunta e aplica estratégias de retrieval e processamento específicas.
"""
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .prompt_templates import QueryType, PromptTemplateManager


@dataclass
class QueryClassification:
    """Resultado da classificação de uma query."""
    query_type: QueryType
    confidence: float
    keywords_matched: List[str]
    reasoning: str
    suggested_filters: Optional[Dict[str, Any]] = None


class QueryRouter:
    """Roteador inteligente que classifica consultas e aplica estratégias apropriadas."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.template_manager = PromptTemplateManager()
        self.patterns = self._create_classification_patterns()
        self.keyword_weights = self._create_keyword_weights()
    
    def _create_classification_patterns(self) -> Dict[QueryType, List[Dict[str, Any]]]:
        """Cria padrões para classificação de consultas."""
        return {
            QueryType.CODE_EXPLANATION: [
                {
                    'patterns': [
                        r'\b(?:explain|what does|how does|meaning of|purpose of)\b.*\b(?:function|method|class|code|implementation)\b',
                        r'\b(?:understand|analyze|break down)\b.*\b(?:code|function|algorithm)\b',
                        r'\bwhat is\b.*\b(?:doing|function|purpose)\b',
                        r'\b(?:how|why)\b.*\b(?:works|implemented|structured)\b'
                    ],
                    'keywords': ['explain', 'what does', 'how does', 'meaning', 'purpose', 'understand', 'analyze', 'works', 'implementation'],
                    'weight': 1.0
                },
                {
                    'patterns': [r'\b(?:function|method|class|variable|parameter)\b.*\b(?:do|does|mean|purpose)\b'],
                    'keywords': ['function', 'method', 'class', 'variable', 'parameter'],
                    'weight': 0.8
                }
            ],
            
            QueryType.CODE_IMPLEMENTATION: [
                {
                    'patterns': [
                        r'\b(?:how to|implement|create|build|write|develop)\b.*\b(?:function|method|class|feature|component)\b',
                        r'\b(?:need to|want to|can I)\b.*\b(?:implement|create|build|add)\b',
                        r'\b(?:example|sample|code)\b.*\b(?:implementation|how to|create)\b',
                        r'\b(?:show|demonstrate|provide)\b.*\b(?:code|implementation|example)\b'
                    ],
                    'keywords': ['implement', 'create', 'build', 'write', 'develop', 'how to', 'example', 'sample'],
                    'weight': 1.0
                },
                {
                    'patterns': [r'\b(?:add|integrate|use)\b.*\b(?:feature|functionality|library|api)\b'],
                    'keywords': ['add', 'integrate', 'use', 'feature', 'functionality'],
                    'weight': 0.8
                }
            ],
            
            QueryType.DEBUGGING: [
                {
                    'patterns': [
                        r'\b(?:error|exception|bug|problem|issue|fail|crash|broken)\b',
                        r'\b(?:not working|doesn\'t work|won\'t work|failing)\b',
                        r'\b(?:debug|fix|solve|resolve|troubleshoot)\b',
                        r'\b(?:why|what\'s wrong|what happened)\b.*\b(?:error|problem|issue)\b'
                    ],
                    'keywords': ['error', 'exception', 'bug', 'problem', 'issue', 'fail', 'debug', 'fix', 'solve', 'troubleshoot'],
                    'weight': 1.0
                },
                {
                    'patterns': [r'\b(?:traceback|stack trace|warning|critical)\b'],
                    'keywords': ['traceback', 'stack trace', 'warning', 'critical'],
                    'weight': 0.9
                }
            ],
            
            QueryType.DOCUMENTATION: [
                {
                    'patterns': [
                        r'\b(?:documentation|docs|guide|tutorial|manual|reference)\b',
                        r'\b(?:how to use|usage|getting started|setup|installation)\b',
                        r'\b(?:what is|overview|introduction|about)\b.*\b(?:project|library|framework|tool)\b',
                        r'\b(?:available|supported|list of)\b.*\b(?:options|features|methods|functions)\b'
                    ],
                    'keywords': ['documentation', 'docs', 'guide', 'tutorial', 'usage', 'setup', 'overview', 'introduction'],
                    'weight': 1.0
                },
                {
                    'patterns': [r'\b(?:readme|getting started|quick start|installation)\b'],
                    'keywords': ['readme', 'getting started', 'installation', 'quick start'],
                    'weight': 0.9
                }
            ],
            
            QueryType.ISSUE_ANALYSIS: [
                {
                    'patterns': [
                        r'\b(?:issue|issues|discussion|thread|conversation)\b.*\b(?:about|regarding|related to)\b',
                        r'\b(?:what was|why was|how was)\b.*\b(?:decided|resolved|implemented|discussed)\b',
                        r'\b(?:pull request|pr|merge)\b.*\b(?:discussion|decision|changes)\b',
                        r'\b(?:consensus|agreement|decision|resolution)\b'
                    ],
                    'keywords': ['issue', 'issues', 'discussion', 'pull request', 'pr', 'consensus', 'decision', 'resolution'],
                    'weight': 1.0
                },
                {
                    'patterns': [r'\b(?:community|contributors|maintainers)\b.*\b(?:think|say|decide)\b'],
                    'keywords': ['community', 'contributors', 'maintainers', 'discussion'],
                    'weight': 0.7
                }
            ],
            
            QueryType.ARCHITECTURE: [
                {
                    'patterns': [
                        r'\b(?:architecture|design|structure|organization|pattern)\b',
                        r'\b(?:how is|how does)\b.*\b(?:organized|structured|designed|architected)\b',
                        r'\b(?:system|application|project)\b.*\b(?:design|structure|architecture)\b',
                        r'\b(?:components|modules|layers|tiers)\b.*\b(?:interact|communicate|organized)\b'
                    ],
                    'keywords': ['architecture', 'design', 'structure', 'organization', 'pattern', 'components', 'modules'],
                    'weight': 1.0
                },
                {
                    'patterns': [r'\b(?:mvc|mvp|microservices|monolithic|layered)\b'],
                    'keywords': ['mvc', 'mvp', 'microservices', 'monolithic', 'layered'],
                    'weight': 0.9
                }
            ],
            
            QueryType.API_USAGE: [
                {
                    'patterns': [
                        r'\b(?:api|endpoint|interface|service)\b.*\b(?:how to|usage|call|invoke)\b',
                        r'\b(?:parameters|arguments|payload|request|response)\b',
                        r'\b(?:authenticate|authorization|token|key)\b.*\b(?:api|service)\b',
                        r'\b(?:rate limit|throttling|quota)\b'
                    ],
                    'keywords': ['api', 'endpoint', 'interface', 'service', 'parameters', 'arguments', 'authenticate', 'rate limit'],
                    'weight': 1.0
                },
                {
                    'patterns': [r'\b(?:rest|graphql|grpc|soap|webhook)\b'],
                    'keywords': ['rest', 'graphql', 'grpc', 'soap', 'webhook'],
                    'weight': 0.8
                }
            ]
        }
    
    def _create_keyword_weights(self) -> Dict[str, float]:
        """Cria pesos para palavras-chave específicas."""
        return {
            # Implementação
            'implement': 1.0, 'create': 0.9, 'build': 0.8, 'develop': 0.8,
            'how to': 1.0, 'example': 0.9, 'sample': 0.8,
            
            # Explicação
            'explain': 1.0, 'what does': 1.0, 'how does': 1.0, 'meaning': 0.8,
            'purpose': 0.8, 'understand': 0.7, 'analyze': 0.7,
            
            # Debug
            'error': 1.0, 'exception': 1.0, 'bug': 1.0, 'problem': 0.9,
            'issue': 0.8, 'fail': 0.9, 'debug': 1.0, 'fix': 0.9,
            
            # Documentação
            'documentation': 1.0, 'docs': 1.0, 'guide': 0.9, 'tutorial': 0.9,
            'usage': 0.8, 'setup': 0.7, 'installation': 0.8,
            
            # Arquitetura
            'architecture': 1.0, 'design': 0.9, 'structure': 0.9,
            'pattern': 0.8, 'organization': 0.7,
            
            # API
            'api': 1.0, 'endpoint': 0.9, 'interface': 0.8, 'service': 0.7,
            'parameters': 0.8, 'authenticate': 0.9
        }
    
    def classify_query(self, query: str) -> QueryClassification:
        """
        Classifica uma consulta determinando seu tipo e características.
        
        Args:
            query: Consulta do usuário
            
        Returns:
            QueryClassification com tipo, confiança e detalhes
        """
        query_lower = query.lower()
        scores = {query_type: 0.0 for query_type in QueryType}
        matched_keywords = {query_type: [] for query_type in QueryType}
        
        # Calcular scores para cada tipo baseado em padrões
        for query_type, pattern_groups in self.patterns.items():
            for group in pattern_groups:
                group_score = 0.0
                group_keywords = []
                
                # Verificar padrões regex
                for pattern in group['patterns']:
                    if re.search(pattern, query_lower, re.IGNORECASE):
                        group_score += group['weight']
                
                # Verificar palavras-chave
                for keyword in group['keywords']:
                    if keyword in query_lower:
                        keyword_weight = self.keyword_weights.get(keyword, 0.5)
                        group_score += keyword_weight * group['weight']
                        group_keywords.append(keyword)
                
                scores[query_type] += group_score
                matched_keywords[query_type].extend(group_keywords)
        
        # Normalizar scores
        max_possible_score = max(scores.values()) if any(scores.values()) else 1.0
        normalized_scores = {qt: score / max_possible_score for qt, score in scores.items()}
        
        # Determinar tipo com maior score
        best_type = max(normalized_scores.items(), key=lambda x: x[1])
        query_type, confidence = best_type
        
        # Se confiança muito baixa, usar GENERAL
        if confidence < 0.1:
            query_type = QueryType.GENERAL
            confidence = 0.5  # Confiança média para general
        
        # Gerar reasoning
        reasoning = self._generate_reasoning(query_type, confidence, matched_keywords[query_type])
        
        # Sugerir filtros baseado no tipo
        suggested_filters = self._suggest_filters(query_type, query_lower)
        
        return QueryClassification(
            query_type=query_type,
            confidence=float(confidence),
            keywords_matched=matched_keywords[query_type],
            reasoning=reasoning,
            suggested_filters=suggested_filters
        )
    
    def _generate_reasoning(self, query_type: QueryType, confidence: float, keywords: List[str]) -> str:
        """Gera explicação para a classificação."""
        reasoning_templates = {
            QueryType.CODE_EXPLANATION: "Detectada solicitação de explicação de código baseada em: {keywords}",
            QueryType.CODE_IMPLEMENTATION: "Identificada necessidade de implementação de código com base em: {keywords}",
            QueryType.DEBUGGING: "Reconhecido problema/erro que requer debugging por: {keywords}",
            QueryType.DOCUMENTATION: "Solicitação de documentação identificada através de: {keywords}",
            QueryType.ISSUE_ANALYSIS: "Consulta sobre discussões/issues detectada por: {keywords}",
            QueryType.ARCHITECTURE: "Pergunta arquitetural identificada via: {keywords}",
            QueryType.API_USAGE: "Consulta sobre uso de API detectada por: {keywords}",
            QueryType.GENERAL: "Classificação geral aplicada (confiança baixa em categorias específicas)"
        }
        
        template = reasoning_templates.get(query_type, "Tipo não reconhecido")
        keywords_str = ", ".join(keywords[:5]) if keywords else "indicadores gerais"
        
        return template.format(keywords=keywords_str) + f" (confiança: {confidence:.2f})"
    
    def _suggest_filters(self, query_type: QueryType, query_lower: str) -> Optional[Dict[str, Any]]:
        """Sugere filtros para otimizar a busca baseado no tipo de consulta."""
        filters = {}
        
        if query_type == QueryType.CODE_EXPLANATION or query_type == QueryType.CODE_IMPLEMENTATION:
            filters['source'] = 'code'
            # Se menciona linguagem específica
            for lang in ['python', 'javascript', 'java', 'go', 'rust', 'cpp']:
                if lang in query_lower:
                    filters['language_detected'] = lang
                    break
        
        elif query_type == QueryType.DEBUGGING:
            # Priorizar issues e código
            filters['source'] = ['issue', 'pull_request', 'code']
        
        elif query_type == QueryType.DOCUMENTATION:
            filters['chunk_type'] = 'documentation'
            # ou arquivo README
            filters['filename_contains'] = 'readme'
        
        elif query_type == QueryType.ISSUE_ANALYSIS:
            filters['source'] = ['issue', 'pull_request']
        
        elif query_type == QueryType.ARCHITECTURE:
            # Incluir múltiplos tipos para visão geral
            pass  # Não filtrar - precisamos de visão ampla
        
        elif query_type == QueryType.API_USAGE:
            filters['source'] = 'code'
            # Procurar por arquivos que podem conter APIs
            filters['has_api_indicators'] = True
        
        return filters if filters else None
    
    def get_optimized_retrieval_params(self, classification: QueryClassification) -> Dict[str, Any]:
        """
        Retorna parâmetros otimizados de retrieval baseado na classificação.
        
        Args:
            classification: Resultado da classificação da query
            
        Returns:
            Dicionário com parâmetros otimizados
        """
        base_params = {
            'top_k_dense': 20,
            'top_k_sparse': 20,
            'final_top_k': 10,
            'dense_weight': 0.7,
            'sparse_weight': 0.3,
            'rerank_enabled': True
        }
        
        # Ajustar baseado no tipo de consulta
        query_type = classification.query_type
        
        if query_type == QueryType.CODE_EXPLANATION:
            # Priorizar busca densa (semântica) para entender código
            base_params.update({
                'dense_weight': 0.8,
                'sparse_weight': 0.2,
                'final_top_k': 8  # Menos documentos, mais focado
            })
        
        elif query_type == QueryType.CODE_IMPLEMENTATION:
            # Equilibrar busca densa e esparsa para encontrar exemplos
            base_params.update({
                'top_k_dense': 25,
                'top_k_sparse': 25,
                'final_top_k': 12,  # Mais exemplos
                'dense_weight': 0.6,
                'sparse_weight': 0.4
            })
        
        elif query_type == QueryType.DEBUGGING:
            # Priorizar busca esparsa para termos específicos de erro
            base_params.update({
                'dense_weight': 0.4,
                'sparse_weight': 0.6,
                'final_top_k': 15,  # Mais contexto para debugging
                'top_k_sparse': 30
            })
        
        elif query_type == QueryType.DOCUMENTATION:
            # Busca equilibrada com foco em documentação estruturada
            base_params.update({
                'dense_weight': 0.7,
                'sparse_weight': 0.3,
                'final_top_k': 8  # Documentação geralmente é mais concisa
            })
        
        elif query_type == QueryType.ISSUE_ANALYSIS:
            # Mais documentos para capturar discussões completas
            base_params.update({
                'top_k_dense': 30,
                'top_k_sparse': 30,
                'final_top_k': 15,
                'dense_weight': 0.6,
                'sparse_weight': 0.4
            })
        
        elif query_type == QueryType.ARCHITECTURE:
            # Muitos documentos para visão ampla do sistema
            base_params.update({
                'top_k_dense': 35,
                'top_k_sparse': 25,
                'final_top_k': 20,
                'dense_weight': 0.8,  # Semântica importante para arquitetura
                'sparse_weight': 0.2
            })
        
        elif query_type == QueryType.API_USAGE:
            # Balanceado com foco em exemplos práticos
            base_params.update({
                'dense_weight': 0.65,
                'sparse_weight': 0.35,
                'final_top_k': 10
            })
        
        # Aplicar filtros se sugeridos
        if classification.suggested_filters:
            base_params['filters'] = classification.suggested_filters
        
        return base_params
    
    def route_query(self, query: str) -> Tuple[QueryClassification, Dict[str, Any]]:
        """
        Rota uma consulta, retornando classificação e parâmetros otimizados.
        
        Args:
            query: Consulta do usuário
            
        Returns:
            Tupla com (classificação, parâmetros de retrieval)
        """
        classification = self.classify_query(query)
        params = self.get_optimized_retrieval_params(classification)
        
        self.logger.info(
            f"Query roteada como {classification.query_type.value} "
            f"(confiança: {classification.confidence:.2f})"
        )
        
        return classification, params