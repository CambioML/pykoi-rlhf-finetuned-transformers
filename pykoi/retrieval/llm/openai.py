"""OpenAI language model for retrieval"""
import os

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from pykoi.retrieval.llm.abs_llm import AbsLlm
from pykoi.retrieval.vectordb.abs_vectordb import AbsVectorDb

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MIN_DOCS = 2


class OpenAIModel(AbsLlm):
    """
    A class representing a language model that uses OpenAI's GPT-3 to generate text.
    """

    def __init__(self, vector_db: AbsVectorDb):
        """
        Initializes the OpenAIModel class.
        """
        try:
            llm = OpenAI(temperature=0, max_tokens=500)

            vector_db = vector_db.vector_db

            retrieve_qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_db.as_retriever(search_kwargs={"k": MIN_DOCS}),
            )

            super().__init__(retrieve_qa)
        except Exception as ex:
            print("Inference initialization failed: {}".format(ex))
