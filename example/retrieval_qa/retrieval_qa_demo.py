"""Demo for the retrieval_qa application."""

import os
import pykoi
import argparse

def main(vector_db_name):
    # enter openai api key here
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["DOC_PATH"] = os.path.join(os.getcwd(), "temp/docs")
    os.environ["VECTORDB_PATH"] = os.path.join(os.getcwd(), "temp/vectordb")
    MODEL_NAME = "openai"

    #####################################
    # Creating a retrieval QA component #
    #####################################
    vector_db = pykoi.VectorDbFactory.create(model_name=MODEL_NAME, vector_db_name=vector_db_name)

    retrieval_model = pykoi.RetrievalFactory.create(
        model_name=MODEL_NAME, vector_db=vector_db
    )

    retriever = pykoi.RetrievalQA(retrieval_model=retrieval_model, vector_db=vector_db)

    ############################################################
    # Starting the application and retrieval qa as a component #
    ############################################################
    # Create the application
    app = pykoi.Application(debug=False, share=False)
    app.add_component(retriever)
    app.run()

if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description="Demo for the Retrieval QA.")
    parser.add_argument("vector_db_name", type=str, help="Name of the vector database")
    args = parser.parse_args()

    # Call the main function with the vector_db_name argument
    main(args.vector_db_name)
