"""Constants for the LLM"""
from enum import Enum


class LlmName(Enum):
    """
    This class is an enumeration for the available Language Learning Models (LLMs).

    Attributes:
        OPENAI (str): Represents the OpenAI model.
        HUGGINGFACE (str): Represents the HuggingFace model.
    """
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
