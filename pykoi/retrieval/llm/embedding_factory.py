"""Embedding factory for LLM"""
from typing import Union

from langchain.embeddings.base import Embeddings
from langchain.embeddings import OpenAIEmbeddings

from pykoi.retrieval.llm.constants import ModelSource


class EmbeddingFactory:
    """
    A factory for creating embeddings.
    """

    @staticmethod
    def create_embedding(model_source: Union[str, ModelSource]) -> Embeddings:
        """
        Create an embedding.

        Args:
            model_source: The name of the model.

        Returns:
            Embeddings: The embedding.
        """
        try:
            model_source = ModelSource(model_source)
            if model_source == ModelSource.OPENAI:
                return OpenAIEmbeddings()
        except Exception as ex:
            raise Exception(f"Unknown embedding: {model_source}") from ex
