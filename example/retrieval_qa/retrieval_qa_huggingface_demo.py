"""Demo for the retrieval_qa application.

python -m example.retrieval_qa.retrieval_qa_huggingface_demo
"""

import os
import argparse
from pykoi import Application
from pykoi.chat import RAGDatabase
from pykoi.retrieval import RetrievalFactory
from pykoi.retrieval import VectorDbFactory
from pykoi.component import Chatbot, Dashboard, RetrievalQA
from dotenv import load_dotenv

# NOTE: Configure your retrieval model as RETRIEVAL_MODEL in .env file.
# Load environment variables from .env file
load_dotenv()

## Set the RETRIEVAL_MODEL, pykoi supports most of the open-source LLMs, e.g.
        # "HuggingFaceH4/zephyr-7b-beta"
        # "meta-llama/Llama-2-7b-chat-hf"
        # "mistralai/Mistral-7B-v0.1"
        # "databricks/dolly-v2-3b" 
        
RETRIEVAL_MODEL = os.getenv("RETRIEVAL_MODEL")


def main(**kwargs):
    os.environ["DOC_PATH"] = os.path.join(os.getcwd(), "temp/docs")
    os.environ["VECTORDB_PATH"] = os.path.join(os.getcwd(), "temp/vectordb")
    MODEL_SOURCE = "huggingface"

    #####################################
    # Creating a retrieval QA component #
    #####################################
    # vector database
    vector_db = VectorDbFactory.create(
        model_source=MODEL_SOURCE,
        vector_db_name=kwargs.get("vectordb"),
        model_name="BAAI/bge-large-en",
        trust_remote_code=True,
        **kwargs
    )

    # retrieval model with vector database
    print("model", RETRIEVAL_MODEL)
    retrieval_model = RetrievalFactory.create(
        model_source=MODEL_SOURCE,
        vector_db=vector_db,
        model_name=RETRIEVAL_MODEL,
        trust_remote_code=True,
        max_length=1000
    )

    # retrieval, chatbot, and dashboard pykoi components
    retriever = RetrievalQA(retrieval_model=retrieval_model, vector_db=vector_db, feedback="rag")
    chatbot = Chatbot(None, feedback="rag", is_retrieval=True)
    dashboard = Dashboard(RAGDatabase(), feedback="rag")

    ############################################################
    # Starting the application and retrieval qa as a component #
    ############################################################
    # Create the application
    app = Application(debug=False, share=False)
    app.add_component(retriever)
    app.add_component(chatbot)
    app.add_component(dashboard)
    print("RUNNING APP IN DEMO MODE")
    app.run()


if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description="Demo for the Retrieval QA.")
    parser.add_argument(
        "-vectordb",
        type=str,
        default="chroma",
        help="Name of the vector database (default: 'chroma')",
    )
    parser.add_argument(
        "-host",
        type=str,
        default="localhost",
        help="Host address if using Epsilla vector database",
    )
    parser.add_argument(
        "-port",
        type=int,
        default=8888,
        help="Port number if using Epsilla vector database",
    )
    args = parser.parse_args()

    # Call the main function with the vector_db_name argument
    main(**vars(args))
