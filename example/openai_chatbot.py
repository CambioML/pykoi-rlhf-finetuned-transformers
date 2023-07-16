import plotano.cambio as cb


# enter openai api key here
api_key = ""

# Creating an OpenAI model
openai_model = cb.ModelFactory.create_model(
    "openai",
    api_key=api_key)

# Creating a chatbot with the OpenAI model
chatbot = cb.Chatbot(openai_model)

# Starting the application (assuming the app is an instance of the Application class)
# Create the application
app = cb.Application()
app.add_component(chatbot)
app.run()
