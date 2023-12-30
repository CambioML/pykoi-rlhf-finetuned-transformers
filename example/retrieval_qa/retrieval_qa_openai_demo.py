"""
Demo for launching a retrieval_qa chatbot UI (with database) from an OpenAI model.

- Prerequisites:
    To run this jupyter notebook, you need a `pykoi` environment with the `rag` option.
    You can follow [the installation guide](https://github.com/CambioML/pykoi/tree/install#option-1-rag-cpu)
    to set up the environment.
- Run the demo:
    1. Enter your OpenAI API key a .env file in the `~/pykoi` directory with the name OPEN_API_KEY, e.g.
        ```
        OPENAI_API_KEY=your_api_key
        ```
    2. On terminal and `~/pykoi` directory, run
        ```
        python -m example.retrieval_qa.retrieval_qa_openai_demo
        ```
"""

import argparse
import os

from dotenv import load_dotenv

from pykoi import Application
from pykoi.chat import RAGDatabase
from pykoi.component import Chatbot, Dashboard, RetrievalQA
from pykoi.retrieval import RetrievalFactory, VectorDbFactory

load_dotenv()


def main(**kargs):
    os.environ["DOC_PATH"] = os.path.join(os.getcwd(), "temp/docs")
    os.environ["VECTORDB_PATH"] = os.path.join(os.getcwd(), "temp/vectordb")
    MODEL_SOURCE = "openai"

    #####################################
    # Creating a retrieval QA component #
    #####################################
    # vector database
    print("1. Creating a vector database...")
    vector_db = VectorDbFactory.create(
        model_source=MODEL_SOURCE, vector_db_name=kargs.get("vectordb"), **kargs
    )
    print("2. Vector database created.")

    # retrieval model with vector database
    retrieval_model = RetrievalFactory.create(
        model_source=MODEL_SOURCE, vector_db=vector_db
    )

    # retrieval, chatbot, and dashboard pykoi components
    retriever = RetrievalQA(
        retrieval_model=retrieval_model, vector_db=vector_db, feedback="rag"
    )
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
