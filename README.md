# GitHub RAG Tool

![GitHub stars](https://img.shields.io/badge/GitHub-RAG-blue)
![Python](https://img.shields.io/badge/Python-3.12.9%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Visão Geral

O GitHub RAG Tool é um projeto dedicado à criação de agentes inteligentes capazes de dialogar com repositórios do GitHub. Utilizando técnicas avançadas de RAG (Retrieval-Augmented Generation), a ferramenta permite que usuários façam perguntas sobre qualquer repositório e obtenham respostas contextualizadas, baseadas no código fonte, issues e pull requests.

## ✨ Funcionalidades

- **Análise Inteligente**: Extrai e indexa informações de repositórios GitHub
- **Resposta Contextual**: Fornece respostas precisas com base no conteúdo do repositório
- **Memória de Conversação**: Mantém o contexto durante toda a interação
- **Rastreabilidade**: Todas as respostas vêm com referências às fontes originais
- **Flexibilidade**: Funciona com qualquer repositório público do GitHub
- **Persistência**: Salva sessões para uso futuro

## 🔧 Requisitos

- Python 3.12.9+
- Chaves de API para serviços de LLM (OpenAI, etc)
- Acesso à internet para conexão com GitHub
- Token de uso da API do GitHub

## 📦 Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/github-rag-tool.git
   cd github-rag-tool
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure o arquivo `.env` com suas credenciais:
   ```
   OPENAI_API_KEY=<sua_chave_openai>
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_EMBBENDING_MODEL=text-embedding-3-large
   GITHUB_API_TOKEN=<seu_token_github>
   ```

## 🛠️ Tecnologias Utilizadas

O projeto utiliza um conjunto de tecnologias modernas para processamento de linguagem natural e recuperação de informações:

- **LangChain**: Framework para construção de aplicações com modelos de linguagem
- **FAISS**: Biblioteca para busca de similaridade eficiente em grandes bases de dados vetoriais
- **OpenAI API**: Fornece modelos como GPT-4o para geração de texto e embeddings
- **GitHub API**: Permite acessar repositórios, issues e pull requests programaticamente
- **Chroma DB**: Banco de dados vetorial para armazenamento de embeddings
- **Python asyncio**: Para processamento assíncrono e melhor desempenho
- **PyTorch**: Base para processamento de modelos de ML (utilizado indiretamente)

## 🔍 Como Usar

### Uso Básico

Execute o script sem argumentos e forneça a URL quando solicitado:

```bash
python main.py
```
### Uso via API Python

```python
from github_rag import SessionManager
import os
from dotenv import load_dotenv

load_dotenv()

config_options = {
    "chunk_size": 1200,
    "chunk_overlap": 300,
    "retriever_k": 7,
    "use_memory": True,
    "memory_window": 5,
}

session_manager = SessionManager(
    repo_url="https://github.com/username/repo",
    initial_config=config_options,
    embeddings_model=os.getenv("OPENAI_EMBBEDDING_MODEL"),
)
session_manager.setup(limit_issues=100, rebuild=False)
result = session_manager.query("Qual é o propósito deste projeto?")
print(result["resposta"])
```

### Fluxo de Trabalho

1. **Inicialização**: A ferramenta inicializa e configura a sessão RAG
2. **Construção da Base de Conhecimento**: O repositório é analisado e indexado
3. **Interação**: Faça perguntas sobre o repositório e receba respostas contextualizadas
4. **Salvamento**: A sessão é automaticamente salva para uso futuro

## ⚙️ Configurações Avançadas

A ferramenta suporta várias configurações para personalizar o comportamento:

```python
config_options = {
    "chunk_size": 1200,       # Tamanho dos trechos de texto para indexação
    "chunk_overlap": 300,     # Sobreposição entre trechos para manter contexto
    "retriever_k": 7,         # Número de documentos recuperados por consulta
    "use_memory": True,       # Habilitar memória de conversação
    "memory_window": 5,       # Tamanho da janela de memória
}
```

## 🔄 Workflow Interno

1. **Extração**: Código, issues e PRs são baixados do repositório
2. **Processamento**: O conteúdo é dividido em chunks significativos
3. **Indexação**: Embeddings são gerados para busca semântica
4. **Recuperação**: Quando uma pergunta é feita, recuperam-se os trechos mais relevantes
5. **Geração**: Um LLM usa os trechos recuperados para produzir respostas precisas
6. **Apresentação**: A resposta é exibida junto com as fontes consultadas

## 📊 Saída de Exemplo

```
🚀 Iniciando sessão com o repositório: https://github.com/huggingface/lerobot

🔧 Inicializando a ferramenta RAG...

⚙️ Configurações aplicadas: {
  "chunk_size": 1200,
  "chunk_overlap": 300,
  "retriever_k": 7,
  "use_memory": true,
  "memory_window": 5
}

🔍 Construindo base de conhecimento...

✅ Preparação concluída em 45.23 segundos

📊 Status da Ferramenta:
- Sessão: session_20250313224434_28d3d4c5
- Modelo de Chat: gpt-4o-mini
- Modelo de Embedding: text-embedding-3-large
- Base vetorial pronta: True
- Documentos indexados: 1724

💬 Modo de consulta ativado para o repositório lerobot
Digite 'sair' para encerrar, 'status' para ver estatísticas, ou 'ajuda' para comandos adicionais

> Qual é o propósito do LeRobot?

🤖 LeRobot visa tornar a IA para robótica mais acessível através da aprendizagem de ponta a ponta. O projeto fornece modelos pré-treinados, datasets e ferramentas para robótica no mundo real usando PyTorch. Seu objetivo é reduzir a barreira de entrada para a robótica, permitindo que todos possam contribuir e se beneficiar do compartilhamento de datasets e modelos pré-treinados.

📚 Fontes:
[1] Arquivo: README.md
    Linguagem: Markdown
[2] Issue #42: Roadmap para implementação de novos ambientes
    URL: https://github.com/huggingface/lerobot/issues/42
```

## 📈 Gerenciamento de Rate Limits

O sistema implementa estratégias inteligentes para lidar com limites de taxa (rate limits) da API OpenAI e GitHub:

- **Retry com backoff exponencial**: Espera progressivamente mais tempo entre tentativas
- **Agrupamento de requisições**: Otimiza o número de chamadas à API
- **Caching de resultados**: Evita requisições redundantes
- **Monitoramento de uso**: Acompanha o consumo de tokens para evitar custos excessivos

## 🔬 Modelos Suportados

A ferramenta é compatível com diversos modelos de linguagem e embeddings:

- **Modelos de Chat**:
  - OpenAI: GPT-4o, GPT-4-Turbo, GPT-3.5-Turbo
  <!-- - Anthropic: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
  - Mistral: Mistral Large, Mistral Medium -->
  
- **Modelos de Embedding**:
  - OpenAI: text-embedding-3-large, text-embedding-3-small, text-embedding-ada-002
  <!-- - Hugging Face: sentence-transformers (via API ou localmente)
  - BAAI: bge-large-en -->

## 🤝 Contribuição

Contribuições são bem-vindas! Para contribuir:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes. 