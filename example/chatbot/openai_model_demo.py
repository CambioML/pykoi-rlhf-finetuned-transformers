"""Demo for the chatbot application using OpenAI endpoint."""
import plotano.cambio as cb

##########################################################
# Creating an OpenAI model (requires an OpenAI API key) #
##########################################################
# enter openai api key here
api_key = ""

# Creating an OpenAI model
model = cb.ModelFactory.create_model(
    model_name="openai",
    api_key=api_key)

#####################################
# Creating a chatbot with the model #
#####################################
database = cb.QuestionAnswerDatabase(debug=True)
chatbot = cb.Chatbot(model=model, feedback="vote")
# chatbot = cb.Chatbot(model=model, feedback="rank")
dashboard = cb.Dashboard(database=database)

###########################################################
# Starting the application and add chatbot as a component #
###########################################################
# Create the application
# app = cb.Application(debug=False, share=True)
app = cb.Application(debug=False, share=False)
app.add_component(chatbot)
app.add_component(dashboard)
app.run()
