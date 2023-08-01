"""Constants for the LLM retriever.""" ""
from enum import Enum


class LlmName(Enum):
    """
    An enum representing the name of a language model.
    """

    OPENAI = "openai"
