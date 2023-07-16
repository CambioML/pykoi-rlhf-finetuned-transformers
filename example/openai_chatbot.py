from plotano.application import Application
from plotano.component.base import Chatbot
from plotano.llm.model_factory import ModelFactory


# enter openai api key here
api_key = "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"

# Creating an OpenAI model
openai_model = ModelFactory.create_model(
    "openai",
    api_key=api_key)

# Creating a chatbot with the OpenAI model
chatbot = Chatbot(openai_model)

# Starting the application (assuming the app is an instance of the Application class)
# Create the application
app = Application()
app.add_component(chatbot)
app.run()
