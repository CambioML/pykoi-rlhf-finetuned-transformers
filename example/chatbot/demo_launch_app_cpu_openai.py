"""
Demo for launching a chatbot UI (with database) from an OpenAI model.

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
        python -m example.chatbot.demo_launch_app_cpu_openai
        ```
"""
import os

from dotenv import load_dotenv

from pykoi import Application
from pykoi.chat import ModelFactory
from pykoi.chat import QuestionAnswerDatabase
from pykoi.component import Chatbot, Dashboard

##########################################################
# Creating an OpenAI model (requires an OpenAI API key) #
##########################################################
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Creating an OpenAI model
model = ModelFactory.create_model(
    model_source="openai",
    api_key=api_key)

#####################################
# Creating a chatbot with the model #
#####################################
database = QuestionAnswerDatabase(debug=True)
chatbot = Chatbot(model=model, feedback="vote")
# chatbot = pykoi.Chatbot(model=model, feedback="rank")
dashboard = Dashboard(database=database)

###########################################################
# Starting the application and add chatbot as a component #
###########################################################
# Create the application
# app = pykoi.Application(debug=False, share=True)
app = Application(debug=False, share=False)
app.add_component(chatbot)
app.add_component(dashboard)
app.run()
