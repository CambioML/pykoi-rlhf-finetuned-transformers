"""Constants for the LLM"""
from enum import Enum


class ModelSource(Enum):
    """
    This class is an enumeration for the available model source.
    """

    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    PEFT_HUGGINGFACE = "peft_huggingface"
