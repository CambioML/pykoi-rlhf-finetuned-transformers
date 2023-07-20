import plotano.cambio as cb


###########################################################
# # Creating an OpenAI model (requires an OpenAI API key) #
###########################################################
# enter openai api key here
api_key = ""

# Creating an OpenAI model
model = cb.ModelFactory.create_model(model_name="openai", api_key=api_key)

###################################################################################
# Creating a Huggingface model tiiuae/falcon-7b (EC2 g5.4xlarge with 100GB space) #
###################################################################################
# model = cb.ModelFactory.create_model(
#     model_name="huggingface",
#     pretrained_model_name_or_path="tiiuae/falcon-7b",
#     trust_remote_code=True,
#     load_in_8bit=True)

#####################################
# Creating a chatbot with the model #
#####################################
database = cb.QuestionAnswerDatabase(debug=True)
chatbot = cb.Chatbot(model=model, database=database, feedback=True)
dashboard = cb.Dashboard(database=database)

###########################################################
# Starting the application and add chatbot as a component #
###########################################################
# Create the application
app = cb.Application(debug=False)
app.add_component(dashboard)
app.run()
