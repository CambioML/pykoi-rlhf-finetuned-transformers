"""OpenAI model wrapper for LLMChain."""
import openai

from plotano.llm.abs_llm import AbsLlm


class OpenAIModel(AbsLlm):
    """OpenAI model wrapper for LLMChain."""

    def __init__(self,
                 api_key: str,
                 engine: str = "davinci",
                 max_tokens: int = 100,
                 temperature: float = 0.5):
        """Initialize the OpenAI model."""
        openai.api_key = api_key
        self._engine = engine
        self._max_tokens = max_tokens
        self._temperature = temperature
        super().__init__()

    def predict(self,
                message: str):
        """Predict the next word."""
        prompt = f"Question: {message}\nAnswer:"
        response = openai.Completion.create(
            engine=self._engine,
            prompt=prompt,
            max_tokens=self._max_tokens,
            n=1,
            stop=None,
            temperature=self._temperature,
        )
        return response.choices[0].text.split("\n")[0]
