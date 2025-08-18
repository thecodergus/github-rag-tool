import os
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class VectorPersistence:
    """
    Gerencia a persistência da base de vetores para sistemas RAG.
    """
    def __init__(
        self,
        embeddings_model: Optional[Any] = None,
        persist_directory: str = "./github_rag_db",
        collection_name: str = "github_data",
    ):
        self.embeddings = embeddings_model or OpenAIEmbeddings()
        self.vector_db = None
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        os.makedirs(persist_directory, exist_ok=True)

    def create_vector_db(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> bool:
        if not documents:
            print("⚠️ Nenhum documento para vetorizar")
            return False
        try:
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            print(f"🔢 Criando vetores para {len(texts)} documentos...")
            if batch_size and len(texts) > batch_size:
                return self._process_in_batches(texts, metadatas, batch_size, show_progress)
            self.vector_db = Chroma.from_texts(
                texts=texts,
                metadatas=metadatas,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
            )
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
        try:
            total_batches = (len(texts) + batch_size - 1) // batch_size
            iterator = range(total_batches)
            if show_progress:
                iterator = tqdm(iterator, desc="Processando batches")
            start_idx = 0
            end_idx = min(batch_size, len(texts))
            self.vector_db = Chroma.from_texts(
                texts=texts[start_idx:end_idx],
                metadatas=metadatas[start_idx:end_idx],
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
            )
            for i in iterator:
                if i == 0:
                    continue
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(texts))
                self.vector_db.add_texts(
                    texts=texts[start_idx:end_idx],
                    metadatas=metadatas[start_idx:end_idx],
                )
                if i % 5 == 0 or i == total_batches - 1:
                    self.vector_db.persist()
            print(f"✅ {len(texts)} documentos processados em {total_batches} batches")
            return True
        except Exception as e:
            print(f"❌ Erro ao processar em batches: {str(e)}")
            return False

    def load_vector_db(self, persist_directory: Optional[str] = None) -> bool:
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
            print(f"✅ Base carregada com sucesso. Contém {collection_size} documentos.")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar base de vetores: {str(e)}")
            return False

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> bool:
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
            if len(texts) > batch_size:
                total_batches = (len(texts) + batch_size - 1) // batch_size
                for i in tqdm(range(total_batches), desc="Adicionando em batches"):
                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, len(texts))
                    self.vector_db.add_texts(
                        texts=texts[start_idx:end_idx],
                        metadatas=metadatas[start_idx:end_idx],
                    )
                    if i % 5 == 0 or i == total_batches - 1:
                        self.vector_db.persist()
            else:
                self.vector_db.add_texts(texts=texts, metadatas=metadatas)
                self.vector_db.persist()
            print(f"✅ {len(texts)} documentos adicionados com sucesso")
            return True
        except Exception as e:
            print(f"❌ Erro ao adicionar documentos: {str(e)}")
            return False

    def delete_collection(self) -> bool:
        if self.vector_db is None:
            print("⚠️ Nenhuma base de vetores inicializada para excluir")
            return False
        try:
            print(f"🗑️ Excluindo coleção {self.collection_name}...")
            self.vector_db._collection.delete(include_metadatas=True)
            print("✅ Coleção excluída com sucesso")
            self.vector_db = None
            return True
        except Exception as e:
            print(f"❌ Erro ao excluir coleção: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        if self.vector_db is None:
            return {"status": "não inicializada"}
        try:
            collection_size = self.vector_db._collection.count()
            metadata_keys = set()
            all_metadatas = self.vector_db._collection.get()["metadatas"]
            for metadata in all_metadatas:
                metadata_keys.update(metadata.keys())
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
                "tipos_fonte": source_types or None,
            }
        except Exception as e:
            return {"status": "erro", "mensagem": str(e)}