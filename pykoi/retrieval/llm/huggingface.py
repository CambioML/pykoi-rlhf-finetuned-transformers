"""OpenAI language model for retrieval"""
import os

from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

from pykoi.retrieval.llm.abs_llm import AbsLlm
from pykoi.retrieval.vectordb.abs_vectordb import AbsVectorDb

MIN_DOCS = 2


class HuggingFaceModel(AbsLlm):
    """
    A class representing a language model that uses OpenAI's GPT-3 to generate text.
    """

    def __init__(self, vector_db: AbsVectorDb, **kwargs):
        """
        Initializes the OpenAIModel class.
        """
        try:
            llm = HuggingFacePipeline.from_model_id(
                model_id=kwargs.get("model_name"),
                task="text-generation",
                model_kwargs={
                    "temperature": 0,
                    "max_length": 500,
                    "trust_remote_code": True,
                },
            )

            vector_db = vector_db.vector_db

            retrieve_qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_db.as_retriever(search_kwargs={"k": MIN_DOCS}),
            )
            print("HuggingFaceModel initialized successfully")
            super().__init__(retrieve_qa)
        except Exception as ex:
            print("Inference initialization failed: {}".format(ex))
