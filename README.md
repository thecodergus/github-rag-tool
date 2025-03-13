# GitHub RAG Tool

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

- Python 3.8+
- Chaves de API para serviços de LLM (OpenAI, etc)
- Acesso à internet para conexão com GitHub

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
   OPENAI_API_KEY=sua_chave_openai
   GITHUB_API_TOKEN=seu_token_github_opcional
   OPENAI_EMBBENDING_MODEL=text-embedding-ada-002
   ```

## 🔍 Como Usar

### Uso Básico

Execute o script principal fornecendo a URL do repositório que deseja analisar:

```bash
python main.py --repo_url https://github.com/username/repo
```

Ou inicie o script sem argumentos e forneça a URL quando solicitado:

```bash
python main.py
# Digite a URL do repositório GitHub: https://github.com/username/repo
```

### Fluxo de Trabalho

1. **Inicialização**: A ferramenta inicializa e configura a sessão RAG
2. **Construção da Base de Conhecimento**: O repositório é analisado e indexado
3. **Interação**: Faça perguntas sobre o repositório e receba respostas contextualizadas
4. **Salvamento**: A sessão é automaticamente salva para uso futuro

### Exemplos de Perguntas

- "Qual é o propósito principal deste repositório?"
- "Como instalo e configuro este projeto?"
- "Quais são as principais dependências?"
- "Explique a arquitetura do sistema"
- "Como posso contribuir para este projeto?"
- "Quais issues estão abertas atualmente?"
- "Mostre-me exemplos de como usar a API"

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
- Documentos indexados: 157
- Tamanho total da base: 25.3 MB
- Tipos de conteúdo: código, issues, pull requests

💬 Faça uma pergunta sobre o repositório (digite 'sair' para encerrar):
> Qual é o propósito do LeRobot?

🤖 LeRobot visa tornar a IA para robótica mais acessível através da aprendizagem de ponta a ponta. O projeto fornece modelos pré-treinados, datasets e ferramentas para robótica no mundo real usando PyTorch. Seu objetivo é reduzir a barreira de entrada para a robótica, permitindo que todos possam contribuir e se beneficiar do compartilhamento de datasets e modelos pré-treinados.

📚 Fontes:
[1] Arquivo: README.md
    Linguagem: Markdown
[2] Issue #42: Roadmap para implementação de novos ambientes
    URL: https://github.com/huggingface/lerobot/issues/42
```

## 🤝 Contribuição

Contribuições são bem-vindas! Para contribuir:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

---

## 🔮 Próximos Passos

- Suporte a repositórios privados
- Interface gráfica para interação mais amigável
- Integração com IDEs populares
- Exportação de relatórios automáticos
- Análise comparativa entre repositórios

---

Desenvolvido com 💙 por Gustavo Michels de Camargo