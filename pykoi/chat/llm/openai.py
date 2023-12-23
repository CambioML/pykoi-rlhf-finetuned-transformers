"""This module provides a wrapper for the OpenAI model."""
from openai import OpenAI

from pykoi.chat.llm.abs_llm import AbsLlm


class OpenAIModel(AbsLlm):
    """
    A class that wraps the OpenAI model for use in the LLMChain.

    Attributes:
        _model (str): The model to use for the OpenAI model.
        _max_tokens (int): The maximum number of tokens to generate.
        _temperature (float): The temperature to use for the OpenAI model.

    Methods:
        __init__(self, model: str, max_tokens: int, temperature: float): Initializes the OpenAI model.
        predict(self, message: str): Predicts the next word based on the given message.
    """

    model_source = "openai"

    def __init__(
        self,
        name: str = None,
        model: str = "davinci",
        max_tokens: int = 100,
        temperature: float = 0.5,
    ):
        """
        Initializes the OpenAI model with the given parameters.

        Args:
            name (str): The name of the model. Defaults to None.
            model (str, optional): The model to use for the OpenAI model. Defaults to "davinci".
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 100.
            temperature (float, optional): The temperature to use for the OpenAI model. Defaults to 0.5.
        """
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._name = name
        self._client = OpenAI()
        super().__init__()

    @property
    def name(self):
        if self._name:
            return self._name
        return "_".join(
            [
                str(OpenAIModel.model_source),
                str(self._model),
                str(self._max_tokens),
                str(self._temperature),
            ]
        )

    def predict(self, message: str, num_of_response: int = 1):
        """
        Predicts the next word based on the given message.

        Args:
            message (str): The message to base the prediction on.
            num_of_response (int): How many completions to generate for each prompt. Defaults to 1.

        Returns:
            List[str]: List of response.
        """
        prompt = f"Question: {message}\nAnswer:"
        response = self._client.completions.create(
            model=self._model,
            prompt=prompt,
            max_tokens=self._max_tokens,
            n=num_of_response,
            stop="\n",
            temperature=self._temperature,
        )
        return [resp.text for resp in response.choices]
