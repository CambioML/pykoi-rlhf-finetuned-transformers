"""Constants for the LLM"""
from enum import Enum


class ModelSource(Enum):
    """
    This class is an enumeration for the available model source.

    Attributes:
        OPENAI (str): Represents the OpenAI model.
        HUGGINGFACE (str): Represents the HuggingFace model.
        PEFT_HUGGINGFACE (str): Represents the PEFT HuggingFace model.
    """

    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    PEFT_HUGGINGFACE = "peft_huggingface"
