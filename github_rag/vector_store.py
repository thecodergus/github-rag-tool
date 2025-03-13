import os
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from tqdm.auto import tqdm
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.retriever import BaseRetriever
from langchain.schema import Document


class VectorStore:
    """
    Gerencia a base de vetores para sistemas RAG (Retrieval Augmented Generation).
    Fornece métodos para criar, carregar, atualizar e consultar bases de dados vetoriais.
    """

    def __init__(
        self,
        embeddings_model: Optional[Any] = None,
        persist_directory: str = "./github_rag_db",
        collection_name: str = "github_data",
    ):
        """
        Inicializa o gerenciador de base vetorial.

        Args:
            embeddings_model: Modelo de embeddings a ser usado (por padrão, OpenAIEmbeddings)
            persist_directory: Diretório onde a base de vetores será persistida
            collection_name: Nome da coleção no Chroma DB
        """
        self.embeddings = embeddings_model or OpenAIEmbeddings()
        self.vector_db = None
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Cria o diretório de persistência, se necessário
        os.makedirs(persist_directory, exist_ok=True)

    def create_vector_db(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> bool:
        """
        Cria base de vetores a partir dos documentos.

        Args:
            documents: Lista de documentos no formato {"text": texto, "metadata": metadados}
            batch_size: Tamanho do lote para processamento em batches
            show_progress: Se deve mostrar barra de progresso

        Returns:
            True se a base foi criada com sucesso, False caso contrário
        """
        if not documents:
            print("⚠️ Nenhum documento para vetorizar")
            return False

        try:
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]

            print(f"🔢 Criando vetores para {len(texts)} documentos...")

            # Processamento em batches para lidar com grandes volumes de dados
            if batch_size and len(texts) > batch_size:
                return self._process_in_batches(
                    texts, metadatas, batch_size, show_progress
                )

            # Processamento direto para conjuntos pequenos
            self.vector_db = Chroma.from_texts(
                texts=texts,
                metadatas=metadatas,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
            )

            # Salva a base explicitamente
            self.vector_db.persist()
            print("✅ Base de vetores criada com sucesso")
            return True

        except Exception as e:
            print(f"❌ Erro ao criar base de vetores: {str(e)}")
            return False

    def _process_in_batches(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int,
        show_progress: bool,
    ) -> bool:
        """
        Processa documentos em batches para evitar problemas de memória.

        Args:
            texts: Lista de textos a serem vetorizados
            metadatas: Lista de metadados correspondentes
            batch_size: Tamanho de cada batch
            show_progress: Se deve mostrar barra de progresso

        Returns:
            True se processado com sucesso, False caso contrário
        """
        try:
            total_batches = (len(texts) + batch_size - 1) // batch_size
            iterator = range(total_batches)

            if show_progress:
                iterator = tqdm(iterator, desc="Processando batches")

            # Processar primeiro lote para inicializar a base
            start_idx = 0
            end_idx = min(batch_size, len(texts))

            self.vector_db = Chroma.from_texts(
                texts=texts[start_idx:end_idx],
                metadatas=metadatas[start_idx:end_idx],
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
            )

            # Processar lotes restantes
            for i in iterator:
                if i == 0:  # Já processamos o primeiro lote
                    continue

                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(texts))

                # Adicionar documentos ao Chroma DB existente
                self.vector_db.add_texts(
                    texts=texts[start_idx:end_idx],
                    metadatas=metadatas[start_idx:end_idx],
                )

                # Persistir após cada lote para evitar perda de dados
                if i % 5 == 0 or i == total_batches - 1:
                    self.vector_db.persist()

            print(f"✅ {len(texts)} documentos processados em {total_batches} batches")
            return True

        except Exception as e:
            print(f"❌ Erro ao processar em batches: {str(e)}")
            return False

    def load_vector_db(self, persist_directory: Optional[str] = None) -> bool:
        """
        Carrega uma base de vetores existente.

        Args:
            persist_directory: Diretório onde a base está armazenada (usa o da instância se None)

        Returns:
            True se carregada com sucesso, False caso contrário
        """
        directory = persist_directory or self.persist_directory

        if not os.path.exists(directory):
            print(f"⚠️ Diretório {directory} não existe")
            return False

        try:
            print(f"📂 Carregando base de vetores de {directory}...")
            self.vector_db = Chroma(
                persist_directory=directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )

            collection_size = self.vector_db._collection.count()
            print(
                f"✅ Base carregada com sucesso. Contém {collection_size} documentos."
            )
            return True

        except Exception as e:
            print(f"❌ Erro ao carregar base de vetores: {str(e)}")
            return False

    def add_documents(
        self, documents: List[Dict[str, Any]], batch_size: int = 100
    ) -> bool:
        """
        Adiciona novos documentos à base de vetores existente.

        Args:
            documents: Lista de documentos no formato {"text": texto, "metadata": metadados}
            batch_size: Tamanho do lote para processamento em batches

        Returns:
            True se documentos foram adicionados com sucesso, False caso contrário
        """
        if not documents:
            print("⚠️ Nenhum documento para adicionar")
            return False

        if self.vector_db is None:
            print("⚠️ Vector database não foi inicializado. Criando novo...")
            return self.create_vector_db(documents, batch_size)

        try:
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]

            print(f"➕ Adicionando {len(texts)} novos documentos à base...")

            # Processamento em batches
            if len(texts) > batch_size:
                total_batches = (len(texts) + batch_size - 1) // batch_size

                for i in tqdm(range(total_batches), desc="Adicionando em batches"):
                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, len(texts))

                    self.vector_db.add_texts(
                        texts=texts[start_idx:end_idx],
                        metadatas=metadatas[start_idx:end_idx],
                    )

                    # Persistir periodicamente
                    if i % 5 == 0 or i == total_batches - 1:
                        self.vector_db.persist()
            else:
                # Adicionar diretamente para conjuntos pequenos
                self.vector_db.add_texts(texts=texts, metadatas=metadatas)
                self.vector_db.persist()

            print(f"✅ {len(texts)} documentos adicionados com sucesso")
            return True

        except Exception as e:
            print(f"❌ Erro ao adicionar documentos: {str(e)}")
            return False

    def get_retriever(
        self, search_kwargs: Optional[Dict[str, Any]] = None
    ) -> BaseRetriever:
        """
        Obtém o retriever da base de vetores com configurações personalizadas.

        Args:
            search_kwargs: Argumentos de pesquisa para o retriever

        Returns:
            Um objeto retriever configurado

        Raises:
            ValueError: Se a base de vetores não foi inicializada
        """
        if self.vector_db is None:
            raise ValueError(
                "Vector database não foi inicializado. Use create_vector_db ou load_vector_db primeiro."
            )

        default_search_kwargs = {"k": 5, "fetch_k": 20, "score_threshold": 0.5}
        if search_kwargs:
            default_search_kwargs.update(search_kwargs)

        return self.vector_db.as_retriever(
            search_type="mmr",  # Usa Maximum Marginal Relevance por padrão
            search_kwargs=default_search_kwargs,
        )

    def query(
        self,
        query_text: str,
        limit: int = 5,
        fetch_k: int = 20,
        include_text: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Realiza uma consulta direta na base de vetores.

        Args:
            query_text: Texto da consulta
            limit: Número máximo de resultados a retornar
            fetch_k: Número de resultados a buscar antes de aplicar MMR
            include_text: Se deve incluir o texto completo nos resultados

        Returns:
            Lista de documentos similares com seus metadados e scores

        Raises:
            ValueError: Se a base de vetores não foi inicializada
        """
        if self.vector_db is None:
            raise ValueError(
                "Vector database não foi inicializado. Use create_vector_db ou load_vector_db primeiro."
            )

        try:
            # Usa MMR (Maximum Marginal Relevance) para diversidade nos resultados
            results = self.vector_db.similarity_search_with_score(
                query=query_text, k=limit, fetch_k=fetch_k
            )

            # Formata os resultados
            formatted_results = []
            for doc, score in results:
                result = {"metadata": doc.metadata, "score": float(score)}

                if include_text:
                    result["text"] = doc.page_content

                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            print(f"❌ Erro ao consultar base de vetores: {str(e)}")
            return []

    def delete_collection(self) -> bool:
        """
        Exclui toda a coleção do Chroma DB.

        Returns:
            True se excluída com sucesso, False caso contrário
        """
        if self.vector_db is None:
            print("⚠️ Nenhuma base de vetores inicializada para excluir")
            return False

        try:
            print(f"🗑️ Excluindo coleção {self.collection_name}...")
            self.vector_db._collection.delete(include_metadatas=True)
            print("✅ Coleção excluída com sucesso")

            # Limpa o objeto vector_db
            self.vector_db = None
            return True

        except Exception as e:
            print(f"❌ Erro ao excluir coleção: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas sobre a base de vetores.

        Returns:
            Dicionário com estatísticas da base
        """
        if self.vector_db is None:
            return {"status": "não inicializada"}

        try:
            collection_size = self.vector_db._collection.count()

            # Obter metadados distintos para estatísticas
            metadata_keys = set()
            all_metadatas = self.vector_db._collection.get()["metadatas"]

            for metadata in all_metadatas:
                metadata_keys.update(metadata.keys())

            # Contar tipos de documentos, se houver campo "source"
            source_types = {}
            if all_metadatas and "source" in all_metadatas[0]:
                for metadata in all_metadatas:
                    source = metadata.get("source", "unknown")
                    source_types[source] = source_types.get(source, 0) + 1

            return {
                "status": "inicializada",
                "caminho": self.persist_directory,
                "coleção": self.collection_name,
                "total_documentos": collection_size,
                "campos_metadados": list(metadata_keys),
                "tipos_fonte": source_types if source_types else None,
            }

        except Exception as e:
            return {"status": "erro", "mensagem": str(e)}
