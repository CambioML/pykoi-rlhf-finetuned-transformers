"""Demo for the chatbot application using OpenAI endpoint."""
from pykoi import Application
from pykoi.chat import ModelFactory
from pykoi.chat import QuestionAnswerDatabase
from pykoi.component import Chatbot, Dashboard


##########################################################
# Creating an OpenAI model (requires an OpenAI API key) #
##########################################################
# enter openai api key here
api_key = ""

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
