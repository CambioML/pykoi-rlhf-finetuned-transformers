"""Chatbot comparator component."""
from typing import List

from pykoi.component.base import Component
from pykoi.component.chatbot_database_factory import ChatbotDatabaseFactory
from pykoi.llm.abs_llm import AbsLlm


class ChatbotComparator(Component):
    def __init__(self, models: List[AbsLlm], **kwargs):
        super().__init__(None, "ChatbotComparator", **kwargs)
        self.models = models
        self.database = ChatbotDatabaseFactory.create(
            feedback=kwargs.get("feedback", "vote")
        )

    def add(self, model: AbsLlm):
        self.models.append(model)
