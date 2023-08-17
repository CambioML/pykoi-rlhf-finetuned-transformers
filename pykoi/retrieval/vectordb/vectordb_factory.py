""" A factory for creating vector databases."""
from typing import Union

from pykoi.retrieval.llm.constants import ModelSource
from pykoi.retrieval.llm.embedding_factory import EmbeddingFactory
from pykoi.retrieval.vectordb.abs_vectordb import AbsVectorDb
from pykoi.retrieval.vectordb.chroma import ChromaDb
from pykoi.retrieval.vectordb.constants import VectorDbName
from pykoi.retrieval.vectordb.epsilla import Epsilla


class VectorDbFactory:
    """
    A factory for creating vector databases.
    """

    @staticmethod
    def create(
        model_source: Union[str, ModelSource],
        vector_db_name: Union[str, VectorDbName],
        **kwargs
    ) -> AbsVectorDb:
        """
        Create a vector database.

        Args:
            model_source: The name of the model.
            vector_db_name: The name of the vector database.
            host: The host address if using Epsilla vector database.
            port: The port number if using Epsilla vector database.

        Returns:
            AbsVectorDb: The vector database.
        """
        try:
            vector_db_name = VectorDbName(vector_db_name)
            model_source = ModelSource(model_source)
            model_embedding = EmbeddingFactory.create_embedding(
                model_source=model_source.value, **kwargs
            )
            if vector_db_name == VectorDbName.CHROMA:
                return ChromaDb(model_embedding)
            if vector_db_name == VectorDbName.EPSILLA:
                return Epsilla(model_embedding, kwargs.get("host"), kwargs.get("port"))

        except Exception as ex:
            raise Exception("Unknown db: {}".format(vector_db_name)) from ex
