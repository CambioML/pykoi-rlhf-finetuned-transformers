"""Model factory for retrieval language models"""
from typing import Union

from pykoi.retrieval.llm.abs_llm import AbsLlm
from pykoi.retrieval.llm.openai import OpenAIModel
from pykoi.retrieval.llm.constants import LlmName
from pykoi.retrieval.vectordb.abs_vectordb import AbsVectorDb


class RetrievalFactory:
    """
    A factory for creating retrieval language models.
    """

    @staticmethod
    def create(model_name: Union[str, LlmName], vector_db: AbsVectorDb) -> AbsLlm:
        """Create a language model for retrieval.

        Args:
            model_name (Union[str, LlmName]): model name
            vector_db (AbsVectorDb): vectoer database

        Returns:
            AbsLlm: Abstract language model for retrieval
        """
        try:
            model_name = LlmName(model_name)
            if model_name == LlmName.OPENAI:
                return OpenAIModel(vector_db)
        except Exception as ex:
            raise Exception(f"Unknown model: {model_name}") from ex
