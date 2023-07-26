"""Module for the Abstract LLM model class."""
import abc


class AbsLlm(abc.ABC):
    """Abstract LLM class.

    This class is an abstract base class (ABC) for LLM classes.
    It ensures that all subclasses implement the `predict` method.

    Attributes:
        None
    """

    @abc.abstractmethod
    def predict(self, message: str, num_of_response: int):
        """Predict the next word based on the input message.

        This method must be implemented by any subclass of `AbsLlm`.

        Args:
            message (str): The input message used to predict the next word.
            num_of_response (int): How many completions to generate for each prompt.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    @property
    def name(self):
        """Return the name of the model.

        This method must be implemented by any subclass of `AbsLlm`.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")
