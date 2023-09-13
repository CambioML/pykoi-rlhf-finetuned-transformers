"""OpenAI language model for retrieval"""
import os

from typing import List

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
            self._llm = OpenAI(
                model_name="gpt-4",
                temperature=0, 
                max_tokens=500)

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

            super().__init__(self._retrieve_qa)
        except Exception as ex:
            print("Inference initialization failed: {}".format(ex))

    def re_init(self, file_names: List[str]):
        """
        Re-initializes the OpenAIModel class.
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

            super().__init__(self._retrieve_qa)
        except Exception as ex:
            print("Inference re-init failed: {}".format(ex))
