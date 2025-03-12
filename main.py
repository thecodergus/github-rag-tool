# Instalação
# pip install 'crewai[tools]' langchain langchain-openai langchain-community pandas python-dotenv chromadb faiss-cpu

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import GithubSearchTool
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory, MongoDBChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pandas as pd
import json
import requests
from datetime import datetime

# Carregar variáveis de ambiente
load_dotenv()

# Configurar tokens
GITHUB_TOKEN = os.getenv("GITHUB_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
MONGODB_URI = os.getenv("MONGODB_URI")


class GitHubRagTool:
    def __init__(
        self,
        repo_url,
        content_types=["code", "issue"],
        custom_model=False,
        session_id=None,
    ):
        """
        Inicializa a ferramenta RAG para GitHub com capacidades de memória.

        Args:
            repo_url: URL do repositório GitHub (ex: 'https://github.com/exemplo/repo')
            content_types: Tipos de conteúdo para busca ('code', 'repo', 'pr', 'issue')
            custom_model: Se True, usa modelos personalizados para embeddings
            session_id: ID da sessão para persistência de memória
        """
        self.repo_url = repo_url
        self.content_types = content_types

        # Gerar ID de sessão se não fornecido
        self.session_id = (
            session_id or f"github_session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )

        # Configurar memória de curto prazo (sessão atual)
        self.short_term_memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # Configurar memória de longo prazo (persistente)
        self.long_term_memory = MongoDBChatMessageHistory(
            connection_string=MONGODB_URI,
            session_id=self.session_id,
            database_name="github_agent_memory",
            collection_name="chat_history",
        )

        # Mesma configuração do GithubSearchTool como no código original
        if custom_model:
            self.tool = GithubSearchTool(
                github_repo=repo_url,
                content_types=content_types,
                gh_token=GITHUB_TOKEN,
                config=dict(
                    llm=dict(
                        provider="openai",
                        config=dict(
                            model=OPENAI_MODEL,
                            temperature=0.5,
                        ),
                    ),
                    embedder=dict(
                        provider="openai",
                        config=dict(
                            model=OPENAI_MODEL,
                        ),
                    ),
                ),
            )
        else:
            self.tool = GithubSearchTool(
                github_repo=repo_url,
                content_types=content_types,
                gh_token=GITHUB_TOKEN,
            )

        # Extrair informações do repositório
        self.owner, self.repo_name = self.parse_repo_url(repo_url)

        # Inicializar a base RAG
        self.vector_db = None
        self.issues_df = None
        self.code_files = None

        # Inicialize a conversation chain e verifique se foi bem-sucedida
        self.conversation_chain = self.create_conversation_chain()

        # Inicializar o modelo de linguagem
        self.llm = ChatOpenAI(model=OPENAI_MODEL)

        self.retriever = self._setup_retriever()

        if self.conversation_chain is None:
            print("ERRO: Falha ao criar a cadeia de conversação!")
            # Log de informações para diagnóstico
            print(f"LLM inicializado: {self.llm is not None}")
            print(f"Retriever inicializado: {self.retriever is not None}")

    def setup_conversation_chain(self):
        """
        Configura a chain de conversação com a base de conhecimento e memória
        """
        if not self.vector_db:
            print("Carregando dados do repositório primeiro...")
            self.load_github_data()

        # Criar a chain de conversação
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            ),
            memory=self.short_term_memory,
            return_source_documents=True,  # Importante para citação de fontes
            verbose=True,
        )

    def load_github_data(self, limit_issues=50, max_files=30):
        """
        Carrega os dados do GitHub e cria a base de conhecimento
        """
        print("Buscando issues...")
        self.issues_df = self.fetch_issues(limit=limit_issues)
        print(f"Encontrados {len(self.issues_df)} issues")

        print("Buscando arquivos de código...")
        self.code_files = self.fetch_code_files(max_files=max_files)
        print(f"Encontrados {len(self.code_files)} arquivos de código")

        # Criar base RAG
        print("Criando base RAG...")
        self.vector_db = self.create_rag_database(self.issues_df, self.code_files)

    def chat(self, user_input):
        if self.conversation_chain is None:
            # Tente criar a chain novamente
            self.conversation_chain = self.create_conversation_chain()

            # Se ainda for None, retorne uma mensagem de erro
            if self.conversation_chain is None:
                return {
                    "answer": "Erro interno: A cadeia de conversação não foi inicializada corretamente. Verifique as configurações e tente novamente.",
                    "sources": [],
                }

        # Continua com o código normal
        result = self.conversation_chain({"question": user_input})
        return result

    def _extract_source_info(self, doc):
        """Extrai informações da fonte do documento"""
        if not doc.metadata:
            return None

        source_path = doc.metadata.get("source", "")

        if "issue_" in source_path:
            # É um issue
            issue_number = source_path.split("issue_")[1].split(".")[0]
            issue_data = self.issues_df[
                self.issues_df["issue_number"] == int(issue_number)
            ]
            if not issue_data.empty:
                return {
                    "type": "issue",
                    "number": issue_number,
                    "title": issue_data.iloc[0]["title"],
                    "url": issue_data.iloc[0]["url"],
                }
        elif "code_" in source_path:
            # É um arquivo de código
            file_name = source_path.split("code_")[1].split(".")[0]
            for file in self.code_files:
                if file["name"].replace("/", "_") == file_name:
                    return {"type": "code", "path": file["path"], "url": file["url"]}

        return None

    def _format_response_with_citations(self, response, sources):
        """Formata a resposta com citações das fontes"""
        if not sources:
            return response

        formatted_response = f"{response}\n\n**Fontes:**\n"

        for i, source in enumerate(sources, 1):
            if source["type"] == "issue":
                formatted_response += f"{i}. Issue #{source['number']}: [{source['title']}]({source['url']})\n"
            elif source["type"] == "code":
                formatted_response += (
                    f"{i}. Arquivo: [{source['path']}]({source['url']})\n"
                )

        return formatted_response

    def parse_repo_url(self, url):
        """Extrai o proprietário e o nome do repositório da URL"""
        parts = url.strip("/").split("/")
        if "github.com" in parts:
            idx = parts.index("github.com")
            if len(parts) > idx + 2:
                return parts[idx + 1], parts[idx + 2]
        raise ValueError(f"URL de repositório inválida: {url}")

    def fetch_issues(self, state="all", limit=100):
        """
        Busca os issues do repositório usando a API do GitHub

        Args:
            state: Estado dos issues ('open', 'closed', 'all')
            limit: Número máximo de issues a serem buscados

        Returns:
            DataFrame com os issues
        """
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        url = f"https://api.github.com/repos/{self.owner}/{self.repo_name}/issues"
        params = {"state": state, "per_page": 100}

        issues = []
        page = 1

        while len(issues) < limit:
            params["page"] = page
            response = requests.get(url, headers=headers, params=params)

            if response.status_code != 200:
                print(f"Erro ao buscar issues: {response.status_code}")
                break

            page_issues = response.json()
            if not page_issues:
                break

            issues.extend(page_issues)
            page += 1

        # Limitar ao número desejado
        issues = issues[:limit]

        # Converter para DataFrame
        df = pd.DataFrame(
            [
                {
                    "issue_number": issue["number"],
                    "title": issue["title"],
                    "state": issue["state"],
                    "created_at": issue["created_at"],
                    "body": issue["body"] if issue["body"] else "",
                    "url": issue["html_url"],
                }
                for issue in issues
            ]
        )

        return df

    def fetch_code_files(self, path="", max_files=50):
        """
        Busca arquivos de código no repositório

        Args:
            path: Caminho dentro do repositório para buscar (vazio = raiz)
            max_files: Número máximo de arquivos a serem buscados

        Returns:
            Lista de informações de arquivos
        """
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        url = f"https://api.github.com/repos/{self.owner}/{self.repo_name}/contents/{path}"

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Erro ao buscar arquivos: {response.status_code}")
            return []

        contents = response.json()
        files = []
        dirs = []

        for item in contents:
            if item["type"] == "file":
                files.append(
                    {
                        "name": item["name"],
                        "path": item["path"],
                        "url": item["html_url"],
                        "download_url": item["download_url"],
                    }
                )
            elif item["type"] == "dir":
                dirs.append(item["path"])

        # Recursivamente buscar em subdiretórios se ainda não atingimos o limite
        if len(files) < max_files and dirs:
            for dir_path in dirs[: min(len(dirs), (max_files - len(files)))]:
                files.extend(self.fetch_code_files(dir_path, max_files - len(files)))
                if len(files) >= max_files:
                    break

        return files[:max_files]

    def download_file_content(self, download_url):
        """Baixa o conteúdo de um arquivo a partir da URL de download"""
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        response = requests.get(download_url, headers=headers)
        if response.status_code == 200:
            return response.text
        return None

    def create_rag_database(self, issues_df, code_files):
        """
        Cria uma base de dados RAG a partir de issues e código.

        Args:
            issues_df: DataFrame com issues
            code_files: Lista de arquivos de código

        Returns:
            ChromaVectorStore: Base de dados vetorial
        """
        print("Criando base RAG...")

        # Garantir que o diretório existe
        os.makedirs("./github_rag_db", exist_ok=True)

        # Processar e salvar issues
        documents = []
        for i, issue in issues_df.iterrows():
            issue_path = f"./github_rag_db/issue_{issue['issue_number']}.txt"
            with open(issue_path, "w", encoding="utf-8") as f:
                f.write(f"Title: {issue['title']}\n")
                f.write(f"State: {issue['state']}\n")
                f.write(f"Created: {issue['created_at']}\n")
                f.write(f"Body:\n{issue['body']}\n")
                f.write(f"URL: {issue['url']}\n")

            # Use UTF-8 encoding when loading the file
            loader = TextLoader(issue_path, encoding="utf-8")
            documents.extend(loader.load())

        # Processar e salvar código
        for file in code_files:
            # Baixar o conteúdo do arquivo
            content = self.download_file_content(file["download_url"])

            # Se conseguiu baixar o conteúdo
            if content:
                file_path = f"./github_rag_db/code_{file['name'].replace('/', '_')}.txt"
                print(file)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"Path: {file['path']}\n")
                    f.write(f"URL: {file['url']}\n")
                    f.write(f"Content:\n{content}\n")

                # Use UTF-8 encoding when loading the file
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
            else:
                print(f"Não foi possível baixar o conteúdo de {file['path']}")

        # Criar embeddings e vectorstore
        embeddings = OpenAIEmbeddings()
        vector_db = Chroma.from_documents(
            documents=documents, embedding=embeddings, persist_directory="./chroma_db"
        )

        return vector_db

    def search_with_tool(self, query):
        """Realiza uma busca semântica usando a ferramenta GithubSearchTool"""
        return self.tool.run(query)

    def create_conversation_chain(self):
        try:
            # Configuração da memória
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key="answer"
            )

            # Criação da cadeia
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=memory,
                return_source_documents=True,
            )

            return conversation_chain
        except Exception as e:
            print(f"ERRO ao criar a chain de conversação: {str(e)}")
            print(f"Tipo de exceção: {type(e).__name__}")
            import traceback

            traceback.print_exc()
            return None

    def _setup_retriever(self):
        """
        Configura e retorna o retriever para busca de informações relevantes.

        Este método verificará se a base de dados vetorial já foi inicializada.
        Se não, retornará None e o retriever será configurado posteriormente
        após o carregamento dos dados do GitHub.

        Returns:
            Retriever ou None: Objeto retriever configurado para busca semântica
        """
        # Se o vector_db ainda não foi inicializado, retorna None
        # Será configurado posteriormente após load_github_data() ser chamado
        if self.vector_db is None:
            print(
                "Base de dados vetorial ainda não inicializada. O retriever será configurado após o carregamento dos dados."
            )
            return None

        # Configurar o retriever a partir da base de dados vetorial
        try:
            # Criar o retriever com busca por similaridade
            retriever = self.vector_db.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5,  # Número de documentos a serem recuperados
                    "fetch_k": 20,  # Busca inicial mais ampla
                    "score_threshold": 0.5,  # Limiar de pontuação para relevância
                },
            )

            print("Retriever configurado com sucesso.")
            return retriever

        except Exception as e:
            print(f"ERRO ao configurar o retriever: {str(e)}")
            import traceback

            traceback.print_exc()
            return None


def conversar_com_repo(repo_url):
    """
    Inicia uma conversa interativa sobre um repositório GitHub

    Args:
        repo_url: URL do repositório GitHub
    """
    print(f"🤖 Iniciando agente para o repositório: {repo_url}")
    print("⏳ Carregando dados e criando base de conhecimento...")

    # Criar o agente com ID de sessão para persistência
    session_id = f"github_{repo_url.split('/')[-2]}_{repo_url.split('/')[-1]}"
    github_agent = GitHubRagTool(
        repo_url=repo_url, content_types=["code", "issue", "pr"], session_id=session_id
    )

    # Carregar dados do GitHub
    github_agent.load_github_data()

    print("\n✅ Agente pronto! Você pode começar a conversar sobre o repositório.")
    print("📝 Digite 'sair' para encerrar a conversa\n")

    while True:
        user_input = input("👤 Você: ")

        if user_input.lower() in ["sair", "exit", "quit"]:
            print("\n👋 Até a próxima!")
            break

        print("\n⏳ Processando...")
        result = github_agent.chat(user_input)

        print(f"\n🤖 Agente: {result['response']}")

        # Se quiser mostrar as fontes separadamente, descomente:
        # if result['sources']:
        #     print("\n📚 Fontes:")
        #     for i, source in enumerate(result['sources'], 1):
        #         if source["type"] == "issue":
        #             print(f"  {i}. Issue #{source['number']}: {source['title']} - {source['url']}")
        #         elif source["type"] == "code":
        #             print(f"  {i}. Arquivo: {source['path']} - {source['url']}")


def main():
    # Exemplo de uso
    repo_url = input("Digite a URL do repositório GitHub: ")

    # Verificar se é uma URL válida
    if "github.com" not in repo_url:
        print("URL inválida. Use o formato: https://github.com/usuario/repositorio")
        return

    conversar_com_repo(repo_url)


if __name__ == "__main__":
    main()
