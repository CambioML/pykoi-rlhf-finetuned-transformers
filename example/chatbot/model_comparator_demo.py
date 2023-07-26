"""Demo for the chatbot application using multiple model endpoint."""
import pykoi


##########################################################
# Creating an OpenAI model (requires an OpenAI API key) #
##########################################################
# enter openai api key here
api_key = ""

# Creating an OpenAI model
openai_model = pykoi.ModelFactory.create_model(
    model_source="openai",
    api_key=api_key)

###################################################################################
# Creating a Huggingface model tiiuae/falcon-7b (EC2 g5.4xlarge with 100GB space) #
###################################################################################
huggingface_model = pykoi.ModelFactory.create_model(
    model_source="huggingface",
    pretrained_model_name_or_path="tiiuae/falcon-7b",
)

#################################
# Creating a chatbot comparator #
#################################
# pass in a list of models to compare
chatbot_comparator = pykoi.ChatbotComparator(
    models=[openai_model])
# or add models later
chatbot_comparator.add(huggingface_model)

app = pykoi.Application(debug=False, share=False)
app.add_component(chatbot_comparator)
app.run()
