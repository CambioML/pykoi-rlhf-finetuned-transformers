from plotano.cambio import Application, Chatbot, ModelFactory

apikey = ""

# Creating an OpenAI model
openai_model = ModelFactory.create_model("openai", api_key=apikey)

# Creating a chatbot with the OpenAI model
chatbot = Chatbot(openai_model)

# Starting the application (assuming the app is an instance of the Application class)
# Create the application
app = Application()
app.add_component(chatbot)
app.run()


# def uppercaseText(text):
#     return text.upper()


# uppercase_model = ModelFactory.create_model(uppercaseText)

# uppercase_chatbot = Chatbot(uppercase_model)

# app = Application()
# app.add_component(uppercase_chatbot)
# app.run()

# Creating a GPT4All model
# gpt4all_model = ModelFactory.create_model(
#     "gpt4all", model_path="orca-mini-3b.ggmlv3.q4_0.bin", max_tokens=3
# )

# # Creating a chatbot with the GPT4All model
# chatbot = Chatbot(gpt4all_model)

# app = Application()
# app.add_component(chatbot)
# app.run()
