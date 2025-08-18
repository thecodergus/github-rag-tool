"""
Templates de prompts especializados para diferentes tipos de consultas RAG.
Otimiza a geração de respostas baseado no contexto e tipo de conteúdo recuperado.
"""
from typing import Dict, List, Any, Optional
from langchain.prompts import PromptTemplate
from enum import Enum


class QueryType(Enum):
    """Tipos de consulta suportados."""
    CODE_EXPLANATION = "code_explanation"
    CODE_IMPLEMENTATION = "code_implementation"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    ISSUE_ANALYSIS = "issue_analysis"
    ARCHITECTURE = "architecture"
    API_USAGE = "api_usage"
    GENERAL = "general"


class PromptTemplateManager:
    """Gerencia templates de prompts especializados para diferentes tipos de consulta."""
    
    def __init__(self):
        self.templates = self._create_templates()
        self.context_enhancers = self._create_context_enhancers()
    
    def _create_templates(self) -> Dict[QueryType, PromptTemplate]:
        """Cria templates especializados para cada tipo de consulta."""
        return {
            QueryType.CODE_EXPLANATION: PromptTemplate.from_template(
                """Você é um especialista em análise de código. Sua tarefa é explicar o código fornecido de forma clara e educativa.

**Contexto do Repositório:**
{context}

**Pergunta:** {question}

**Instruções:**
1. Analise o código fornecido no contexto
2. Explique o propósito e funcionamento do código
3. Identifique padrões, bibliotecas e frameworks utilizados  
4. Destaque aspectos importantes da implementação
5. Se relevante, mencione possíveis melhorias ou alternativas
6. Use exemplos práticos quando apropriado
7. Seja preciso e técnico, mas mantenha clareza

**Formato da Resposta:**
- Começe com um resumo do que o código faz
- Explique os componentes principais
- Detalhe a lógica de implementação
- Conclua com insights relevantes

**Resposta:**"""
            ),
            
            QueryType.CODE_IMPLEMENTATION: PromptTemplate.from_template(
                """Você é um desenvolvedor sênior especializado em implementação de código. Forneça soluções práticas e funcionais.

**Contexto do Repositório:**
{context}

**Pergunta:** {question}

**Instruções:**
1. Analise o contexto fornecido para entender o padrão do projeto
2. Implemente código que seja consistente com o estilo existente
3. Use as mesmas bibliotecas e frameworks já presentes no projeto
4. Forneça código completo e funcional
5. Inclua comentários explicativos quando necessário
6. Considere tratamento de erros e casos extremos
7. Sugira testes se apropriado

**Formato da Resposta:**
```linguagem
// Código implementado aqui
```

**Explicação:**
- Explique as decisões de design
- Destaque integração com código existente
- Mencione dependências necessárias

**Resposta:**"""
            ),
            
            QueryType.DEBUGGING: PromptTemplate.from_template(
                """Você é um especialista em debugging e resolução de problemas. Analise o problema e forneça soluções práticas.

**Contexto do Repositório:**
{context}

**Pergunta:** {question}

**Instruções:**
1. Identifique o problema descrito na pergunta
2. Analise o código/contexto fornecido em busca de possíveis causas
3. Considere erros comuns relacionados ao tipo de problema
4. Forneça soluções ordenadas por probabilidade de sucesso
5. Explique por que cada solução pode resolver o problema
6. Inclua métodos de diagnóstico e debugging
7. Sugira práticas para prevenir problemas similares

**Formato da Resposta:**
**Problema Identificado:** [Resumo do problema]

**Possíveis Causas:**
1. [Causa mais provável]
2. [Segunda causa mais provável]
...

**Soluções Recomendadas:**
1. **[Solução principal]**
   - Implementação: [código/passos]
   - Por que funciona: [explicação]

**Métodos de Diagnóstico:**
- [Como verificar se a solução funcionou]

**Resposta:**"""
            ),
            
            QueryType.DOCUMENTATION: PromptTemplate.from_template(
                """Você é um especialista técnico em documentação de software. Forneça informações claras e bem estruturadas.

**Contexto do Repositório:**
{context}

**Pergunta:** {question}

**Instruções:**
1. Use o contexto fornecido para responder de forma precisa
2. Organize a informação de forma hierárquica e lógica
3. Inclua exemplos práticos quando apropriado
4. Destaque informações importantes ou requisitos especiais
5. Se a informação estiver incompleta no contexto, indique claramente
6. Forneça referências para documentação adicional quando possível
7. Use formatação clara com cabeçalhos, listas e código quando necessário

**Formato da Resposta:**
Use markdown para estruturar a resposta com:
- Cabeçalhos para organizar seções
- Listas para enumerar itens
- Blocos de código para exemplos
- Links para recursos adicionais

**Resposta:**"""
            ),
            
            QueryType.ISSUE_ANALYSIS: PromptTemplate.from_template(
                """Você é um analista de issues especializado em projetos de software. Analise discussões e forneça insights.

**Contexto do Repositório:**
{context}

**Pergunta:** {question}

**Instruções:**
1. Analise as issues/PRs fornecidas no contexto
2. Identifique temas e padrões nas discussões
3. Resuma decisões importantes e consensos alcançados
4. Destaque soluções propostas e suas implementações
5. Identifique problemas recorrentes ou não resolvidos
6. Extraia lições aprendidas e boas práticas
7. Se relevante, relacione com outras issues similares

**Formato da Resposta:**
**Resumo:** [Visão geral do tópico]

**Pontos Principais:**
- [Decisões importantes]
- [Soluções implementadas]
- [Problemas identificados]

**Status Atual:** [Estado da discussão/implementação]

**Insights:** [Lições aprendidas ou recomendações]

**Resposta:**"""
            ),
            
            QueryType.ARCHITECTURE: PromptTemplate.from_template(
                """Você é um arquiteto de software especializado em análise de sistemas. Forneça insights arquiteturais profundos.

**Contexto do Repositório:**
{context}

**Pergunta:** {question}

**Instruções:**
1. Analise a estrutura e organização do código fornecido
2. Identifique padrões arquiteturais utilizados
3. Avalie decisões de design e suas implicações
4. Destaque pontos fortes e possíveis melhorias
5. Considere aspectos de escalabilidade, manutenibilidade e performance
6. Relacione com melhores práticas da indústria
7. Forneça recomendações específicas quando apropriado

**Formato da Resposta:**
**Visão Arquitetural:** [Descrição da arquitetura atual]

**Padrões Identificados:**
- [Padrão 1]: [Descrição e uso]
- [Padrão 2]: [Descrição e uso]

**Pontos Fortes:**
- [Aspecto positivo 1]
- [Aspecto positivo 2]

**Áreas de Melhoria:**
- [Sugestão 1]: [Benefícios]
- [Sugestão 2]: [Benefícios]

**Resposta:**"""
            ),
            
            QueryType.API_USAGE: PromptTemplate.from_template(
                """Você é um especialista em APIs e integrações. Forneça orientação prática sobre uso de APIs.

**Contexto do Repositório:**
{context}

**Pergunta:** {question}

**Instruções:**
1. Identifique a API ou interface em questão
2. Forneça exemplos práticos de uso
3. Explique parâmetros, retornos e comportamentos
4. Destaque melhores práticas de implementação
5. Inclua tratamento de erros apropriado
6. Considere aspectos de performance e limitações
7. Forneça código funcional sempre que possível

**Formato da Resposta:**
**API:** [Nome/identificação da API]

**Uso Básico:**
```linguagem
// Exemplo de uso básico
```

**Parâmetros:**
- `parametro1`: [descrição e tipo]
- `parametro2`: [descrição e tipo]

**Retorno:** [Descrição do que a API retorna]

**Exemplo Completo:**
```linguagem
// Exemplo completo com tratamento de erros
```

**Considerações:**
- [Melhores práticas]
- [Limitações importantes]

**Resposta:**"""
            ),
            
            QueryType.GENERAL: PromptTemplate.from_template(
                """Você é um assistente especializado em desenvolvimento de software e análise de código.

**Contexto do Repositório:**
{context}

**Pergunta:** {question}

**Instruções:**
1. Analise cuidadosamente o contexto fornecido
2. Responda de forma precisa baseado nas informações disponíveis
3. Se a informação não estiver completa no contexto, indique isso claramente
4. Use exemplos do próprio repositório quando possível
5. Seja técnico mas mantenha clareza na explicação
6. Forneça informações práticas e acionáveis
7. Se apropriado, sugira próximos passos ou recursos adicionais

**Resposta:**"""
            )
        }
    
    def _create_context_enhancers(self) -> Dict[QueryType, callable]:
        """Cria funções para enriquecer o contexto baseado no tipo de consulta."""
        return {
            QueryType.CODE_EXPLANATION: self._enhance_code_context,
            QueryType.CODE_IMPLEMENTATION: self._enhance_implementation_context,
            QueryType.DEBUGGING: self._enhance_debugging_context,
            QueryType.DOCUMENTATION: self._enhance_documentation_context,
            QueryType.ISSUE_ANALYSIS: self._enhance_issue_context,
            QueryType.ARCHITECTURE: self._enhance_architecture_context,
            QueryType.API_USAGE: self._enhance_api_context,
            QueryType.GENERAL: self._enhance_general_context
        }
    
    def get_template(self, query_type: QueryType) -> PromptTemplate:
        """Retorna o template apropriado para o tipo de consulta."""
        return self.templates.get(query_type, self.templates[QueryType.GENERAL])
    
    def enhance_context(self, 
                       context_docs: List[Dict[str, Any]], 
                       query_type: QueryType,
                       query: str) -> str:
        """
        Enriquece o contexto baseado no tipo de consulta.
        
        Args:
            context_docs: Documentos recuperados
            query_type: Tipo da consulta
            query: Query original
            
        Returns:
            Contexto enriquecido formatado
        """
        enhancer = self.context_enhancers.get(query_type, self._enhance_general_context)
        return enhancer(context_docs, query)
    
    def _enhance_code_context(self, context_docs: List[Dict[str, Any]], query: str) -> str:
        """Enriquece contexto para explicação de código."""
        enhanced_parts = []
        
        for i, doc in enumerate(context_docs, 1):
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            # Adicionar informações do arquivo
            filename = metadata.get('filename', 'arquivo_desconhecido')
            language = metadata.get('language_detected', 'unknown')
            
            enhanced_parts.append(f"""
**Documento {i} - {filename}**
Linguagem: {language}
```{language if language != 'unknown' else ''}
{text}
```
""")
        
        return "\n".join(enhanced_parts)
    
    def _enhance_implementation_context(self, context_docs: List[Dict[str, Any]], query: str) -> str:
        """Enriquece contexto para implementação de código."""
        enhanced_parts = []
        code_examples = []
        dependencies = set()
        
        for doc in context_docs:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            # Extrair imports/dependencies
            if 'import ' in text or 'from ' in text:
                imports = [line.strip() for line in text.split('\n') 
                          if line.strip().startswith(('import ', 'from '))]
                dependencies.update(imports)
            
            # Separar código de documentação
            if metadata.get('source') == 'code':
                code_examples.append({
                    'filename': metadata.get('filename', ''),
                    'text': text,
                    'language': metadata.get('language_detected', 'unknown')
                })
        
        # Adicionar dependências identificadas
        if dependencies:
            enhanced_parts.append("**Dependencies/Imports identificados:**")
            for dep in sorted(dependencies)[:10]:  # Limitar a 10
                enhanced_parts.append(f"- {dep}")
            enhanced_parts.append("")
        
        # Adicionar exemplos de código
        enhanced_parts.append("**Exemplos de código do projeto:**")
        for i, example in enumerate(code_examples[:5], 1):  # Limitar a 5 exemplos
            enhanced_parts.append(f"""
**Exemplo {i} - {example['filename']}**
```{example['language'] if example['language'] != 'unknown' else ''}
{example['text'][:1000]}{'...' if len(example['text']) > 1000 else ''}
```
""")
        
        return "\n".join(enhanced_parts)
    
    def _enhance_debugging_context(self, context_docs: List[Dict[str, Any]], query: str) -> str:
        """Enriquece contexto para debugging."""
        enhanced_parts = []
        error_related = []
        code_snippets = []
        
        for doc in context_docs:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            # Identificar conteúdo relacionado a erros
            if any(keyword in text.lower() for keyword in ['error', 'exception', 'bug', 'fix', 'problem']):
                error_related.append({
                    'text': text,
                    'type': metadata.get('source', 'unknown'),
                    'filename': metadata.get('filename', '')
                })
            elif metadata.get('source') == 'code':
                code_snippets.append({
                    'text': text,
                    'filename': metadata.get('filename', ''),
                    'language': metadata.get('language_detected', 'unknown')
                })
        
        # Adicionar discussões relacionadas a erros
        if error_related:
            enhanced_parts.append("**Discussões sobre problemas similares:**")
            for item in error_related[:3]:
                source_type = "Issue/PR" if item['type'] in ['issue', 'pull_request'] else "Documentação"
                enhanced_parts.append(f"""
**{source_type}**: {item['filename']}
{item['text'][:500]}{'...' if len(item['text']) > 500 else ''}
""")
        
        # Adicionar código relevante
        if code_snippets:
            enhanced_parts.append("\n**Código relacionado:**")
            for snippet in code_snippets[:3]:
                enhanced_parts.append(f"""
**Arquivo**: {snippet['filename']}
```{snippet['language'] if snippet['language'] != 'unknown' else ''}
{snippet['text'][:800]}{'...' if len(snippet['text']) > 800 else ''}
```
""")
        
        return "\n".join(enhanced_parts)
    
    def _enhance_documentation_context(self, context_docs: List[Dict[str, Any]], query: str) -> str:
        """Enriquece contexto para documentação."""
        enhanced_parts = []
        
        # Organizar por tipo de documento
        docs_by_type = {'readme': [], 'docs': [], 'code_comments': [], 'other': []}
        
        for doc in context_docs:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            filename = metadata.get('filename', '').lower()
            
            if 'readme' in filename:
                docs_by_type['readme'].append((text, metadata))
            elif metadata.get('extension') in ['.md', '.rst', '.txt']:
                docs_by_type['docs'].append((text, metadata))
            elif metadata.get('source') == 'code' and ('"""' in text or "'''" in text or '//' in text):
                docs_by_type['code_comments'].append((text, metadata))
            else:
                docs_by_type['other'].append((text, metadata))
        
        # Adicionar seções organizadas
        for doc_type, docs in docs_by_type.items():
            if docs:
                type_name = {
                    'readme': 'README e Documentação Principal',
                    'docs': 'Documentação Adicional', 
                    'code_comments': 'Comentários e Docstrings do Código',
                    'other': 'Outras Fontes'
                }[doc_type]
                
                enhanced_parts.append(f"**{type_name}:**")
                for text, metadata in docs[:2]:  # Limitar a 2 por tipo
                    filename = metadata.get('filename', 'fonte_desconhecida')
                    enhanced_parts.append(f"""
*Fonte*: {filename}
{text[:800]}{'...' if len(text) > 800 else ''}
""")
                enhanced_parts.append("")
        
        return "\n".join(enhanced_parts)
    
    def _enhance_issue_context(self, context_docs: List[Dict[str, Any]], query: str) -> str:
        """Enriquece contexto para análise de issues."""
        enhanced_parts = []
        issues = []
        prs = []
        
        for doc in context_docs:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            source = metadata.get('source', '')
            
            if source == 'issue':
                issues.append((text, metadata))
            elif source == 'pull_request':
                prs.append((text, metadata))
        
        # Adicionar issues
        if issues:
            enhanced_parts.append("**Issues Relacionadas:**")
            for text, metadata in issues[:3]:
                number = metadata.get('number', 'N/A')
                title = metadata.get('title', 'Sem título')
                enhanced_parts.append(f"""
**Issue #{number}**: {title}
{text[:600]}{'...' if len(text) > 600 else ''}
""")
        
        # Adicionar pull requests
        if prs:
            enhanced_parts.append("\n**Pull Requests Relacionados:**")
            for text, metadata in prs[:3]:
                number = metadata.get('number', 'N/A')
                title = metadata.get('title', 'Sem título')
                enhanced_parts.append(f"""
**PR #{number}**: {title}
{text[:600]}{'...' if len(text) > 600 else ''}
""")
        
        return "\n".join(enhanced_parts)
    
    def _enhance_architecture_context(self, context_docs: List[Dict[str, Any]], query: str) -> str:
        """Enriquece contexto para análise arquitetural."""
        enhanced_parts = []
        file_structure = {}
        
        # Organizar por estrutura de arquivos
        for doc in context_docs:
            metadata = doc.get('metadata', {})
            filename = metadata.get('filename', '')
            text = doc.get('text', '')
            
            if filename:
                # Extrair diretório
                directory = '/'.join(filename.split('/')[:-1]) if '/' in filename else 'root'
                if directory not in file_structure:
                    file_structure[directory] = []
                file_structure[directory].append((filename, text, metadata))
        
        enhanced_parts.append("**Estrutura do Projeto:**")
        for directory, files in sorted(file_structure.items())[:5]:  # Limitar diretórios
            enhanced_parts.append(f"\n**Diretório**: `{directory}`")
            for filename, text, metadata in files[:3]:  # Limitar arquivos por diretório
                language = metadata.get('language_detected', 'unknown')
                enhanced_parts.append(f"""
- **{filename.split('/')[-1]}** ({language})
  ```{language if language != 'unknown' else ''}
  {text[:300]}{'...' if len(text) > 300 else ''}
  ```
""")
        
        return "\n".join(enhanced_parts)
    
    def _enhance_api_context(self, context_docs: List[Dict[str, Any]], query: str) -> str:
        """Enriquece contexto para uso de APIs."""
        enhanced_parts = []
        api_examples = []
        
        for doc in context_docs:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            # Procurar por definições de função/API
            if any(keyword in text for keyword in ['def ', 'function', 'class ', 'async ', 'api']):
                api_examples.append({
                    'text': text,
                    'filename': metadata.get('filename', ''),
                    'language': metadata.get('language_detected', 'unknown')
                })
        
        enhanced_parts.append("**Definições de API/Funções encontradas:**")
        for i, example in enumerate(api_examples[:4], 1):
            enhanced_parts.append(f"""
**Exemplo {i}** - {example['filename']}
```{example['language'] if example['language'] != 'unknown' else ''}
{example['text'][:600]}{'...' if len(example['text']) > 600 else ''}
```
""")
        
        return "\n".join(enhanced_parts)
    
    def _enhance_general_context(self, context_docs: List[Dict[str, Any]], query: str) -> str:
        """Enriquecimento geral de contexto."""
        enhanced_parts = []
        
        for i, doc in enumerate(context_docs, 1):
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            # Informações básicas do documento
            source_info = []
            if metadata.get('filename'):
                source_info.append(f"Arquivo: {metadata['filename']}")
            if metadata.get('source'):
                source_info.append(f"Tipo: {metadata['source']}")
            if metadata.get('language_detected'):
                source_info.append(f"Linguagem: {metadata['language_detected']}")
            
            source_line = " | ".join(source_info) if source_info else "Fonte desconhecida"
            
            enhanced_parts.append(f"""
**Documento {i}**
*{source_line}*

{text[:800]}{'...' if len(text) > 800 else ''}
""")
        
        return "\n".join(enhanced_parts)