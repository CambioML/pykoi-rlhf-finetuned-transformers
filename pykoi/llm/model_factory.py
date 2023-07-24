"""This module defines a factory for creating language models."""

from typing import Union

from pykoi.llm.abs_llm import AbsLlm
from pykoi.llm.constants import ModelSource


class ModelFactory:
    """
    A factory class for creating language models.

    This class provides a static method `create_model` which creates a
    language model instance based on the given name.

    Methods:
        create_model(model_source: Union[str, ModelSource], **kwargs) -> AbsLlm:
            Creates a language model based on the given name.
    """

    @staticmethod
    def create_model(model_source: Union[str, ModelSource], **kwargs) -> AbsLlm:
        """
        Create a language model based on the given name.

        This method tries to match the given model name with the names defined
        in the `ModelSource` enumeration. If a match is found, it creates an
        instance of the corresponding language model. If no match is found,
        it raises a ValueError.

        Args:
            model_source (Union[str, ModelSource]): The name of the language model
                source.

        Returns:
            AbsLlm: An instance of the language model.

        Raises:
            ValueError: If the given model name is not valid.
        """
        try:
            model_source = ModelSource(model_source)
            if model_source == ModelSource.OPENAI:
                from pykoi.llm.openai import OpenAIModel

                return OpenAIModel(**kwargs)
            elif model_source == ModelSource.HUGGINGFACE:
                from pykoi.llm.huggingface import HuggingfaceModel

                return HuggingfaceModel(**kwargs)
            elif model_source == ModelSource.PEFT_HUGGINGFACE:
                from pykoi.llm.peft_huggingface import PeftHuggingfacemodel

                return PeftHuggingfacemodel(**kwargs)
            else:
                raise ValueError(f"[llm_factory]: Unknown model source " f"{model_source}")
        except ValueError as ex:
            raise ValueError("[llm_factory]: initialize model failure") from ex
