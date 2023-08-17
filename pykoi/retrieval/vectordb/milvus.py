import os
import numpy as np

from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Milvus
from sklearn.decomposition import PCA

from pykoi.retrieval.vectordb.abs_vectordb import AbsVectorDb


class MilvusDb(AbsVectorDb):
    def __init__(self, embedding: Embeddings):
        self._embedding = embedding
        print("before init")
        self._vector_db = Milvus(
            embedding_function=self._embedding,
            collection_name="milvus_collection",
            connection_args={"host": "127.0.0.1", "port": 19530},
        )
        print("after init")
        super().__init__()

    def _get_file_names(self):
        return set([])

    def _index(self, texts, metadatas):
        self._vector_db.add_texts(texts=texts, metadatas=metadatas)

    def _persist(self):
        pass

    def get_embedding(self):
        pass
