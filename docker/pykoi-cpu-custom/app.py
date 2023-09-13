"""Demo for the chatbot application using OpenAI endpoint."""
import pykoi

##########################################################
# Creating an OpenAI model (requires an OpenAI API key) #
##########################################################
# enter openai api key here
api_key = "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"

# Creating an OpenAI model
model = pykoi.ModelFactory.create_model(
    model_source="openai",
    api_key=api_key)

#####################################
# Creating a chatbot with the model #
#####################################
database = pykoi.QuestionAnswerDatabase(debug=True)
chatbot = pykoi.Chatbot(model=model, feedback="vote")
# chatbot = pykoi.Chatbot(model=model, feedback="rank")
dashboard = pykoi.Dashboard(database=database)

###########################################################
# Starting the application and add chatbot as a component #
###########################################################
# Create the application
# app = pykoi.Application(debug=False, share=True)
app = pykoi.Application(
    debug=False, 
    share=True)
app.add_component(chatbot)
app.add_component(dashboard)
app.run()
