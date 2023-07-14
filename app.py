from cambio import Chatbot, Application


# Define a function that represents your model
def uppercase_model(text):
    return text.upper()


chatbot = Chatbot("Chatbot", uppercase_model, feedback=False)

app = Application()

app.add_component(chatbot)

app.run()
