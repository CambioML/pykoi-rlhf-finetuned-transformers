"""OpenAI language model for retrieval"""
import os
import torch

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
                device=torch.cuda.device_count() - 1,
                pipeline_kwargs={"device_map": "auto"},
                model_kwargs={
                    "temperature": 0,
                    "max_length": kwargs.get("max_length", 500),
                    "trust_remote_code": kwargs.get("trust_remote_code", True),
                    # "load_in_8bit": True,
                },
            )

            vector_db = vector_db.vector_db

            retrieve_qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_db.as_retriever(search_kwargs={"k": MIN_DOCS}),
                verbose=True,
            )
            print("HuggingFaceModel initialized successfully")
            super().__init__(retrieve_qa)
        except Exception as ex:
            print("Inference initialization failed: {}".format(ex))
