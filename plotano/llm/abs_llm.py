"""Abstract LLM model class."""
import abc


class AbsLlm(abc.ABC):
    """Abstract LLM class."""

    @abc.abstractmethod
    def predict(self, message: str):
        """Predict the next word."""
        raise NotImplementedError("This method must be implemented by subclasses.")
