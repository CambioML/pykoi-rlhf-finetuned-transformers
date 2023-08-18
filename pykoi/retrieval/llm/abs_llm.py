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
        print("before get_relevant_documents")
        result = self._retrieve_qa.retriever.get_relevant_documents(message)
        print(result)
        print("after get_relevant_documents")
        return self._retrieve_qa.run(message)
