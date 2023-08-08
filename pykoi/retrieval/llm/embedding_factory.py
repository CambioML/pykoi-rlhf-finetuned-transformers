"""Embedding factory for LLM"""
from typing import Union

from langchain.embeddings.base import Embeddings
from langchain.embeddings import OpenAIEmbeddings

from pykoi.retrieval.llm.constants import LlmName


class EmbeddingFactory:
    """
    A factory for creating embeddings.
    """

    @staticmethod
    def create_embedding(model_name: Union[str, LlmName]) -> Embeddings:
        """
        Create an embedding.

        Args:
            model_name: The name of the model.

        Returns:
            Embeddings: The embedding.
        """
        try:
            model_name = LlmName(model_name)
            if model_name == LlmName.OPENAI:
                return OpenAIEmbeddings()
        except Exception as ex:
            raise Exception(f"Unknown embedding: {model_name}") from ex
