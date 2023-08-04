import os
import numpy as np
import types

from typing import List, Union
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document
from sklearn.decomposition import PCA
from pyepsilla import vectordb

from pykoi.retrieval.llm.constants import LlmName
from pykoi.retrieval.vectordb.abs_vectordb import AbsVectorDb

model_map = {
    LlmName.OPENAI: 1536
}

class EpsillaRetriever(BaseRetriever):
    def __init__(self, vector_db, embedding, limit):
        self.vector_db = vector_db
        self.embedding = embedding
        self.limit = limit

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self.embedding.embed_query(query)
        status_code, response = self.vector_db.query(
            table_name="RetrievalQA",
            query_field="embeddings",
            query_vector=query_vector,
            limit=self.limit
        )
        if status_code != 200:
            print(f"Error: {response['message']}.")
            raise Exception("Error: {}.".format(response['message']))

        return list(map(lambda item: Document(page_content = item["text"], metadata = dict()), response["result"]))

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        # This is a placeholder for the asynchronous version of get_relevant_documents
        # The actual implementation would depend on how the vectordb.Client and the embedding object handle
        # asynchronous operations
        pass

class Epsilla(AbsVectorDb):
    def __init__(self, model_name: Union[str, LlmName], embedding: Embeddings):
        self._embedding = embedding
        self._vector_db = vectordb.Client(
            host="localhost",
            port=8888,
        )

        def as_retriever(search_kwargs) -> BaseRetriever:
            limit = search_kwargs["k"]
            return EpsillaRetriever(self._vector_db, self._embedding, limit)

        def as_retriever_wrapper(self, search_kwargs):
            return as_retriever(search_kwargs)

        self._vector_db.as_retriever = types.MethodType(as_retriever_wrapper, self._vector_db)

        self._vector_db.load_db(db_name="PykoiDB", db_path="/tmp/pykoi")
        self._vector_db.use_db(db_name="PykoiDB")
        result = self._vector_db.create_table(
            table_name="RetrievalQA",
            table_fields=[
            {"name": "file_name", "dataType": "STRING"},
            {"name": "text", "dataType": "STRING"},
            {"name": "embeddings", "dataType": "VECTOR_FLOAT", "dimensions": model_map[model_name]}
            ]
        )
        if (result[1]['message'] == "Table already exists: RetrievalQA"):
            print("Table RetrievalQA already exists. Continuing with the existing table.")
        super().__init__()

    def _get_file_names(self):
        status_code, response = self._vector_db.get(table_name="RetrievalQA", response_fields=["file_name"])
        if (status_code != 200):
            print(f"Error: {response['message']}.")
            raise Exception("Error: {}.".format(response['message']))
        else:
            return set(metadata_dict["file_name"] for metadata_dict in response["result"])

    def _index(self, texts, metadatas):
        embeddings = self._embedding.embed_documents(texts)
        records = []
        for index, metadata_dict in enumerate(metadatas):
            record = {
                "file_name": metadata_dict["file_name"],
                "text": texts[index],
                "embeddings": embeddings[index]
            }
            records.append(record)
        self._vector_db.insert(table_name="RetrievalQA", records=records)

    def _persist(self):
        pass

    def get_embedding(self):
        status_code, response = self._vector_db.get(table_name="RetrievalQA", response_fields=["file_name", "embeddings"])
        if (status_code != 200):
            print(f"Error: {response['message']}.")
            raise Exception("Error: {}.".format(response['message']))

        embedding_list = [result_dict["embeddings"] for result_dict in response["result"]]
        file_name_list = [result_dict["file_name"] for result_dict in response["result"]]

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
