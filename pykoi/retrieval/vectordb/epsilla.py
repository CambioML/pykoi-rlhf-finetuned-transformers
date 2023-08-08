"""Vector store Epsilla module"""
import os
import numpy as np
import types

from typing import List
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document
from langchain.embeddings import OpenAIEmbeddings
from sklearn.decomposition import PCA
from pyepsilla import vectordb

from pykoi.retrieval.vectordb.abs_vectordb import AbsVectorDb


class EpsillaRetriever(BaseRetriever):
    """Epsilla retriever class."""

    def __init__(self, vector_db, embedding, limit):
        self.vector_db = vector_db
        self.embedding = embedding
        self.limit = limit

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        query_vector = self.embedding.embed_query(query)
        status_code, response = self.vector_db.query(
            table_name="RetrievalQA",
            query_field="embeddings",
            query_vector=query_vector,
            limit=self.limit,
        )
        if status_code != 200:
            print(f"Error: {response['message']}.")
            raise Exception("Error: {}.".format(response["message"]))

        return list(
            map(
                lambda item: Document(page_content=item["text"], metadata=dict()),
                response["result"],
            )
        )

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        # This is a placeholder for the asynchronous version of get_relevant_documents
        # The actual implementation would depend on how the vectordb.Client and the embedding object handle
        # asynchronous operations
        pass


class Epsilla(AbsVectorDb):
    """Vector store Epsilla class"""

    def __init__(self, embedding: Embeddings, host: str, port: int):
        """
        Initializes a new instance of the Epsilla class.
        Connect to Epsilla vector database.
        Creat/load and use DB "PykoiDB" and create/use table "RetrievalQA" with fields:
            - file_name: STRING
            - text: STRING
            - embeddings: VECTOR_FLOAT (with corresponding dimensions for different model name)

        Args:
            embedding (Ebeddings): The embedding.
            host (str): The host address of the Epsilla vector database. Default is "localhost".
            port (int): The port number of the Epsilla vector database. Default is 8888.
        """
        if not isinstance(embedding, OpenAIEmbeddings):
            raise TypeError(
                "Invalid type for 'embedding'. Expected OpenAIEmbeddings instance."
            )
        if embedding.model == "text-embedding-ada-002":
            dimensions = 1536
        else:
            raise NotImplementedError(f"Unsupported embedding model: {embedding.model}")

        self._embedding = embedding
        self._vector_db = vectordb.Client(
            host=host,
            port=port,
        )

        def as_retriever(search_kwargs) -> BaseRetriever:
            """Return a base retriever used by Epsilla vector database."""
            limit = search_kwargs["k"]
            return EpsillaRetriever(self._vector_db, self._embedding, limit)

        def as_retriever_wrapper(self, search_kwargs):
            """Wrapper for as_retriever function."""
            return as_retriever(search_kwargs)

        self._vector_db.as_retriever = types.MethodType(
            as_retriever_wrapper, self._vector_db
        )

        self._vector_db.load_db(
            db_name="PykoiDB", db_path="{}/epsilla".format(os.getenv("VECTORDB_PATH"))
        )
        self._vector_db.use_db(db_name="PykoiDB")
        result = self._vector_db.create_table(
            table_name="RetrievalQA",
            table_fields=[
                {"name": "file_name", "dataType": "STRING"},
                {"name": "text", "dataType": "STRING"},
                {
                    "name": "embeddings",
                    "dataType": "VECTOR_FLOAT",
                    "dimensions": dimensions,
                },
            ],
        )
        if result[1]["message"] == "Table already exists: RetrievalQA":
            print(
                "Table RetrievalQA already exists. Continuing with the existing table."
            )
        super().__init__()

    def _get_file_names(self):
        """Return a set of file names."""
        status_code, response = self._vector_db.get(
            table_name="RetrievalQA", response_fields=["file_name"]
        )
        if status_code != 200:
            print(f"Error: {response['message']}.")
            raise Exception("Error: {}.".format(response["message"]))
        else:
            return set(
                metadata_dict["file_name"] for metadata_dict in response["result"]
            )

    def _index(self, texts, metadatas):
        """
        Create embeddings for a list of text.
        Inserts the embeddings with file name and original text into database.

        Args:
            texts (List[str]): The texts to insert.
            metadatas (List[dict]): The file names to insert.
        """
        embeddings = self._embedding.embed_documents(texts)
        records = []
        for index, metadata_dict in enumerate(metadatas):
            record = {
                "file_name": metadata_dict["file_name"],
                "text": texts[index],
                "embeddings": embeddings[index],
            }
            records.append(record)
        status_code, response = self._vector_db.insert(
            table_name="RetrievalQA", records=records
        )
        if status_code != 200:
            print(f"Error: {response['message']}.")
            raise Exception("Error: {}.".format(response["message"]))

    def _persist(self):
        """Placeholder for _persist function in abstract vectordb class."""
        pass

    def get_embedding(self):
        """
        Retrieves embeddings from the vector database and performs PCA dimensionality reduction.
        Returns a dictionary containing the PCA results, label indices, and file names.

        Returns:
            Dict[str, Union[List[int], List[str], List[List[float]]]]: A dictionary with the following keys:
                - 'labels': A list of integers representing unique indices for each file name.
                - 'labelNames': A list of strings containing the unique file names.
                - 'projection': A list of lists of floats representing the PCA projection results.
                                Each inner list corresponds to an embedded data point.
        """
        status_code, response = self._vector_db.get(
            table_name="RetrievalQA", response_fields=["file_name", "embeddings"]
        )
        if status_code != 200:
            print(f"Error: {response['message']}.")
            raise Exception("Error: {}.".format(response["message"]))

        embedding_list = [
            result_dict["embeddings"] for result_dict in response["result"]
        ]
        file_name_list = [
            result_dict["file_name"] for result_dict in response["result"]
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
