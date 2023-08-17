import os
import numpy as np

from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from sklearn.decomposition import PCA

from pykoi.retrieval.vectordb.abs_vectordb import AbsVectorDb


class ChromaDb(AbsVectorDb):
    def __init__(self, embedding: Embeddings):
        self._embedding = embedding
        self._vector_db = Chroma(
            persist_directory="{}/chroma".format(os.getenv("VECTORDB_PATH")),
            embedding_function=self._embedding,
        )
        super().__init__()

    def _get_file_names(self):
        vector_db_list = self._vector_db.get(include=["metadatas"])
        return set(
            [
                metadata_dict["file_name"]
                for metadata_dict in vector_db_list["metadatas"]
            ]
        )

    def _index(self, texts, metadatas):
        self._vector_db.add_texts(texts=texts, metadatas=metadatas)

    def _persist(self):
        self._vector_db.persist()

    def get_embedding(self):
        vector_db_list = self._vector_db.get(include=["embeddings", "metadatas"])
        embedding_list = vector_db_list["embeddings"]
        file_name_list = [
            metadata_dict["file_name"] for metadata_dict in vector_db_list["metadatas"]
        ]
        if not embedding_list or min(len(embedding_list), len(embedding_list[0])) < 3:
            index_list = []
            file_name_list = []
            pca_result_list = []
            print("Not enough datapoint to run PCA, upload and index more files first.")
        else:
            # do a PCA dimensionality reduction
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(np.array(embedding_list))
            pca_result_list = pca_result.tolist()

            # create a dictionary of unique labels
            unique_indices = {
                item: index for index, item in enumerate(set(file_name_list))
            }
            index_list = [unique_indices[item] for item in file_name_list]

        return {
            "labels": index_list,
            "labelNames": file_name_list,
            "projection": pca_result_list,
        }
