"""Chatbot Database Factory class."""
from typing import Union

from pykoi.component.constants import FeedbackType
from pykoi.chat.db.qa_database import QuestionAnswerDatabase
from pykoi.chat.db.ranking_database import RankingDatabase
from pykoi.chat.db.rag_database import RAGDatabase


class ChatbotDatabaseFactory:
    """
    A factory class for creating chatbot databases.
    """

    @staticmethod
    def create(feedback: Union[str, FeedbackType]):
        """
        Create a chatbot database.

        Args:
            feedback (Union[str, FeedbackType]): The type of the chatbot.

        Returns:
            Union[QuestionAnswerDatabase, RankingDatabase, RAGDatabase]: The created database.
        """
        feedback = FeedbackType(feedback)
        if feedback == FeedbackType.VOTE:
            return QuestionAnswerDatabase()
        elif feedback == FeedbackType.RANK:
            return RankingDatabase()
        elif feedback == FeedbackType.RAG:
            return RAGDatabase()
        else:
            raise ValueError(
                f"Invalid feedback name: {feedback}. "
                "Valid values are: 'question_answer', 'ranking'."
            )
