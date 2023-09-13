"""Model factory for retrieval language models"""
from typing import Union

from pykoi.retrieval.llm.abs_llm import AbsLlm
from pykoi.retrieval.llm.constants import ModelSource
from pykoi.retrieval.vectordb.abs_vectordb import AbsVectorDb


class RetrievalFactory:
    """
    A factory for creating retrieval language models.
    """

    @staticmethod
    def create(
        model_source: Union[str, ModelSource], vector_db: AbsVectorDb, **kwargs
    ) -> AbsLlm:
        """Create a language model for retrieval.

        Args:
            model_source (Union[str, LlmName]): model name
            vector_db (AbsVectorDb): vector database

        Returns:
            AbsLlm: Abstract language model for retrieval
        """
        try:
            model_source = ModelSource(model_source)
            if model_source == ModelSource.OPENAI:
                from pykoi.retrieval.llm.openai import OpenAIModel
                return OpenAIModel(vector_db)
            if model_source == ModelSource.HUGGINGFACE:
                from pykoi.retrieval.llm.huggingface import HuggingFaceModel
                return HuggingFaceModel(vector_db, **kwargs)
        except Exception as ex:
            raise Exception(f"Unknown model: {model_source}") from ex
