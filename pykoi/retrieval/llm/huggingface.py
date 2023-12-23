"""OpenAI language model for retrieval"""
import os

import torch
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

from pykoi.retrieval.llm.abs_llm import AbsLlm
from pykoi.retrieval.vectordb.abs_vectordb import AbsVectorDb

# NOTE: Configure your MIN_DOCS as RAG_NUM_SOURCES in .env file.
# Load environment variables from .env file
load_dotenv()

MIN_DOCS = int(os.getenv("RAG_NUM_SOURCES", default=2))


class HuggingFaceModel(AbsLlm):
    """
    A class representing a language model that uses Huggingface's model to generate text.
    """

    def __init__(self, vector_db: AbsVectorDb, **kwargs):
        """
        Initializes the HuggingFaceModel class.
        """
        try:
            self._llm = HuggingFacePipeline.from_model_id(
                model_id=kwargs.get("model_name"),
                task="text-generation",
                device=torch.cuda.device_count() - 1,
                # pipeline_kwargs={"device_map": "auto"},
                model_kwargs={
                    "temperature": 0,
                    "max_length": kwargs.get("max_length", 500),
                    "load_in_8bit": True,
                    "trust_remote_code": kwargs.get("trust_remote_code", True),
                },
            )

            self._vector_db = vector_db.vector_db

            self._retrieve_qa = RetrievalQA.from_chain_type(
                llm=self._llm,
                chain_type="stuff",
                retriever=self._vector_db.as_retriever(
                    search_kwargs={"k": MIN_DOCS, "filter": {}}
                ),
                verbose=True,
                return_source_documents=True,
            )
            print("HuggingFaceModel initialized successfully!")
            super().__init__(self._retrieve_qa)
        except Exception as ex:
            print("Inference initialization failed: {}".format(ex))

    def re_init(self, file_names: list[str]):
        """
        Re-initializes the HuggingFaceModel class.
        """
        try:
            if file_names == []:
                metadata_filename_filter = {"file_name": ""}
            elif len(file_names) == 1:
                metadata_filename_filter = {"file_name": file_names[0]}
            else:
                metadata_filename_filter = {
                    "$or": [{"file_name": name} for name in file_names]
                }

            self._retrieve_qa = RetrievalQA.from_chain_type(
                llm=self._llm,
                chain_type="stuff",
                retriever=self._vector_db.as_retriever(
                    search_kwargs={"k": MIN_DOCS, "filter": metadata_filename_filter}
                ),
                verbose=True,
                return_source_documents=True,
            )

            print(
                "Re-initialized HuggingFaceModel successfully with filter: ",
                metadata_filename_filter,
            )

            super().__init__(self._retrieve_qa)
        except Exception as ex:
            print("Inference re-init failed: {}".format(ex))
