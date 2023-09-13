"""Embedding factory for LLM"""
from typing import Union

from langchain.embeddings.base import Embeddings
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

from pykoi.retrieval.llm.constants import ModelSource


class EmbeddingFactory:
    """
    A factory for creating embeddings.
    """

    @staticmethod
    def create_embedding(model_source: Union[str, ModelSource], **kwargs) -> Embeddings:
        """
        Create an embedding.

        Args:
            model_source: The name of the model.
            **kwargs: Keyword arguments for the model.

        Returns:
            Embeddings: The embedding.
        """
        try:
            model_source = ModelSource(model_source)
            if model_source == ModelSource.OPENAI:
                from langchain.embeddings import OpenAIEmbeddings
                return OpenAIEmbeddings()
            elif model_source == ModelSource.HUGGINGFACE:
                from langchain.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(
                    model_name=kwargs.get("model_name"),
                )
        except Exception as ex:
            raise Exception(f"Unknown embedding: {model_source}") from ex
