"""Constants for the LLM"""
from enum import Enum


class LlmName(Enum):
    """Enumeration of the available LLMs."""
    LLAMA_CPP = "llama_cpp"
    OPENAI = "openai"
    SAGEMAKER = "sagemaker"
