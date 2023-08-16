"""Constants for the LLM retriever.""" ""
from enum import Enum


class ModelSource(Enum):
    """
    An enum representing the name of a language model.
    """

    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
