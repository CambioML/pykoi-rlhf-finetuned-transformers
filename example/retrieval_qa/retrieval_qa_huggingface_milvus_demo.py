"""Demo for the retrieval_qa application.

for development without pip install the package, run the following command in the root directory: 
python -m example.retrieval_qa.retrieval_qa_huggingface_milvus_demo

for development after pip install the package, run the following command in the root directory:
python example/retrieval_qa/retrieval_qa_huggingface_milvus_demo.py
"""

import os
import argparse
from milvus import default_server
from pymilvus import connections, utility

import pykoi


def main(**kwargs):
    os.environ["DOC_PATH"] = os.path.join(os.getcwd(), "temp/docs")
    os.environ["VECTORDB_PATH"] = os.path.join(os.getcwd(), "temp/vectordb")
    MODEL_SOURCE = "huggingface"

    with default_server:
        # TODO: set base dir to your milvus path
        default_server.set_base_dir("{}/milvus".format(os.getenv("VECTORDB_PATH")))
        connections.connect(host=kwargs.get("host"), port=kwargs.get("port"))
        print(utility.get_server_version())
        #####################################
        # Creating a retrieval QA component #
        #####################################
        # vector database
        print("Creating vector database...")
        vector_db = pykoi.VectorDbFactory.create(
            model_source=MODEL_SOURCE,
            vector_db_name=kwargs.get("vectordb"),
            model_name="BAAI/bge-large-en",
            trust_remote_code=True,
            **kwargs
        )
        # retrieval model with vector database
        print("Creating retrieval model...")
        retrieval_model = pykoi.RetrievalFactory.create(
            model_source=MODEL_SOURCE,
            vector_db=vector_db,
            model_name="databricks/dolly-v2-3b",
            trust_remote_code=True,
            max_length=1000,
        )

        # retrieval and chatbot components
        retriever = pykoi.RetrievalQA(
            retrieval_model=retrieval_model, vector_db=vector_db
        )
        chatbot = pykoi.Chatbot(None, feedback="vote", is_retrieval=True)
        database = pykoi.QuestionAnswerDatabase(debug=True)
        dashboard = pykoi.Dashboard(database=database)

        ############################################################
        # Starting the application and retrieval qa as a component #
        ############################################################
        # Create the application
        app = pykoi.Application(debug=False, share=False)
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
        default="milvus",
        help="Name of the vector database (default: 'chroma')",
    )
    parser.add_argument(
        "-host",
        type=str,
        default="127.0.0.1",
        help="Host address if using Epsilla vector database",
    )
    parser.add_argument(
        "-port",
        type=int,
        default=19530,
        help="Port number if using Epsilla vector database",
    )
    args = parser.parse_args()

    # Call the main function with the vector_db_name argument
    main(**vars(args))
