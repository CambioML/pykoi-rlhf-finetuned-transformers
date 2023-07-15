"""Abstract LLM model class."""
from langchain.llms.base import BaseLLM


class AbsLlm():
    """Abstract LLM class."""

    def __init__(self, model: BaseLLM):
        """Initialize the LLM model."""
        self._model = model

    def predict(self, text):
        """Predict the next word."""
        return self._model.predict(text)
