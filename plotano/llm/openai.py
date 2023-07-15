"""OpenAI model wrapper for LLMChain."""
from langchain import OpenAI

from plotano.llm.abs_llm import AbsLlm


class OpenAIModel(AbsLlm):
    """OpenAI model wrapper for LLMChain."""

    def __init__(self):
        """Initialize the OpenAI model."""
        model = OpenAI(temperature=0)
        super().__init__(model=model)
