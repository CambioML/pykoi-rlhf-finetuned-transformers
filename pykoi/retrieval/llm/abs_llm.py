"""Abstract class for retrieval based language learning model."""
from abc import ABC

from langchain.chains import RetrievalQA


class AbsLlm(ABC):
    """
    Abstract class for retrieval based language learning model.
    """

    def __init__(self, retrieve_qa: RetrievalQA) -> None:
        """
        Constructor for the AbsLlm class.

        Args:
            retrieve_qa (RetrievalQA): The retrieval question answering model.
        """
        self._retrieve_qa = retrieve_qa

    def run(self, message: str) -> str:
        """
        Runs the language learning model.

        Args:
            message (str): The message to run the model on.

        Returns:
            str: The response from the model.
        """
        return self._retrieve_qa.run(message)

    def run_with_return_source_documents(self, message: dict) -> dict:
        """
        Runs the language learning model, return source documents

        Args:
            message (dict): The message dict to run the model on.

        Returns:
            dict: The response dict from the model.
        """
        return self._retrieve_qa(message)
