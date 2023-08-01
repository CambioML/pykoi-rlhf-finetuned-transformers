"""Demo for the retrieval_qa application."""

import os
import pykoi


# enter openai api key here
os.environ["OPENAI_API_KEY"] = ""
os.environ["DOC_PATH"] = os.path.join(os.getcwd(), "temp/docs")
os.environ["VECTORDB_PATH"] = os.path.join(os.getcwd(), "temp/vectordb")
MODEL_NAME = "openai"

#####################################
# Creating a retrieval QA component #
#####################################
vector_db = pykoi.VectorDbFactory.create(model_name=MODEL_NAME, vector_db_name="chroma")

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
