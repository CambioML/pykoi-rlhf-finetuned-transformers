"""This module defines a factory for creating language models."""

from typing import Union

from plotano.llm.constants import LlmName
from plotano.llm.abs_llm import AbsLlm


class ModelFactory:
    """
    A factory class for creating language models.

    This class provides a static method `create_model` which creates a 
    language model instance based on the given name.

    Methods:
        create_model(model_name: Union[str, LlmName], **kwargs) -> AbsLlm: 
            Creates a language model based on the given name.
    """
    @staticmethod
    def create_model(model_name: Union[str, LlmName],
                     **kwargs) -> AbsLlm:
        """
        Create a language model based on the given name.

        This method tries to match the given model name with the names defined 
        in the `LlmName` enumeration. If a match is found, it creates an 
        instance of the corresponding language model. If no match is found, 
        it raises a ValueError.

        Args:
            model_name (Union[str, LlmName]): The name of the language model.

        Returns:
            AbsLlm: An instance of the language model.

        Raises:
            ValueError: If the given model name is not valid.
        """
        try:
            model_name = LlmName(model_name)
            if model_name == LlmName.OPENAI:
                from plotano.llm.openai import OpenAIModel

                return OpenAIModel(**kwargs)
            elif model_name == LlmName.HUGGINGFACE:
                from plotano.llm.huggingface import HuggingfaceModel

                return HuggingfaceModel(**kwargs)
            else:
                raise ValueError(f"[llm_factory]: Unknown model "
                                 f"{model_name}")
        except ValueError as ex:
            raise ValueError("[llm_factory]: initialize model failure") from ex
