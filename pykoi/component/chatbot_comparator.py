"""Chatbot comparator component."""
from typing import List

from pykoi.component.base import Component
from pykoi.component.chatbot_database_factory import ChatbotDatabaseFactory
from pykoi.llm.abs_llm import AbsLlm


class ChatbotComparator(Component):
    """Chatbot comparator component."""
    def __init__(self, models: List[AbsLlm], **kwargs):
        """
        Initializes a new instance of the ChatbotComparator class.

        Args:
            models (List[AbsLlm]): A list of models to compare.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            Exception: If a model with the same name already exists.

        """
        super().__init__(None, "ChatbotComparator", **kwargs)
        self.models = {}
        for model in models:
            self.add(model)
        self.database = ChatbotDatabaseFactory.create(
            feedback=kwargs.get("feedback", "vote")
        )

    def add(self, model: AbsLlm):
        """
        Adds a model to the comparator.

        Args:
            model (AbsLlm): The model to add.

        Raises:
            Exception: If a model with the same name already exists.
        """
        if model.name in self.models:
            raise Exception(f"Model {model.name} already exists")
        self.models[model.name] = model
