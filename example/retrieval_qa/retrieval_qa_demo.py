"""Demo for the retrieval_qa application."""

import os
import argparse
import pykoi


def main(**kargs):
    # enter openai api key here
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["DOC_PATH"] = os.path.join(os.getcwd(), "temp/docs")
    os.environ["VECTORDB_PATH"] = os.path.join(os.getcwd(), "temp/vectordb")
    MODEL_SOURCE = "openai"

    #####################################
    # Creating a retrieval QA component #
    #####################################
    # vector database
    vector_db = pykoi.VectorDbFactory.create(
        model_source=MODEL_SOURCE, vector_db_name=kargs.get("vectordb"), **kargs
    )

    # retrieval model with vector database
    retrieval_model = pykoi.RetrievalFactory.create(
        model_source=MODEL_SOURCE, vector_db=vector_db
    )

    # sql database
    database = pykoi.QuestionAnswerDatabase(debug=True)
    dashboard = pykoi.Dashboard(database=database)

    # Creating an OpenAI model
    model = pykoi.ModelFactory.create_model(
        model_source=MODEL_SOURCE, api_key=os.environ["OPENAI_API_KEY"]
    )

    # retrieval and chatbot components
    retriever = pykoi.RetrievalQA(retrieval_model=retrieval_model, vector_db=vector_db)
    chatbot = pykoi.Chatbot(model=model, feedback="vote", is_retrieval=True)

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
