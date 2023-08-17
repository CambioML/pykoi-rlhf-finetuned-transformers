"""Retrieval QA component."""

from pykoi.retrieval.llm.abs_llm import AbsLlm
from pykoi.retrieval.vectordb.abs_vectordb import AbsVectorDb
from pykoi.component.chatbot_database_factory import ChatbotDatabaseFactory

from pykoi.component.base import Component


class RetrievalQA(Component):
    """Retrieval QA component."""

    def __init__(self, retrieval_model: AbsLlm, vector_db: AbsVectorDb, **kwargs):
        """
        Initializes a new instance of the RetrievalQA class.

        Args:
            retrieval_model (AbsLlm): The retrieval_model to use.
            vector_db (AbsVectorDb): The vector database to use.
            **kwargs: Arbitrary keyword arguments.

        """
        super().__init__(None, "RetrievalQA", **kwargs)
        self.retrieval_model = retrieval_model
        self.vector_db = vector_db
        self.database = ChatbotDatabaseFactory.create(
            feedback=kwargs.get("feedback", "vote")
        )
