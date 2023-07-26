"""Demo for the chatbot application using multiple model endpoint."""
import pykoi


##########################################################
# Creating an OpenAI model (requires an OpenAI API key) #
##########################################################
# enter openai api key here
api_key = ""

# Creating an OpenAI model
openai_model_1 = pykoi.ModelFactory.create_model(
    model_source="openai",
    api_key=api_key,
    engine="babbage")
openai_model_2 = pykoi.ModelFactory.create_model(
    model_source="openai",
    api_key=api_key,
    engine="curie")
openai_model_3 = pykoi.ModelFactory.create_model(
    model_source="openai",
    api_key=api_key,
    engine="davinci")

###################################################################################
# Creating a Huggingface model tiiuae/falcon-7b (EC2 g5.4xlarge with 100GB space) #
###################################################################################
# huggingface_model = pykoi.ModelFactory.create_model(
#     model_source="huggingface",
#     pretrained_model_name_or_path="tiiuae/falcon-7b",
# )

###################################################################################
# Creating a Huggingface model tiiuae/falcon-7b (EC2 g5.4xlarge with 100GB space) #
###################################################################################
# only run this after run rlfh/sft_demo.py to fine tune the model
# peft_huggingface_model = pykoi.ModelFactory.create_model(
#     model_source="peft_huggingface",
#     base_model_path="meta-llama/Llama-2-7b-hf",
#     lora_model_path="/home/ubuntu/pykoi/models/rlhf_step1_sft",
# )

#################################
# Creating a chatbot comparator #
#################################
# pass in a list of models to compare
chatbot_comparator = pykoi.ChatbotComparator(
    models=[openai_model_1, openai_model_2])
chatbot_comparator.add(openai_model_3)
# or add models later
# chatbot_comparator.add(huggingface_model)
# chatbot_comparator.add(peft_huggingface_model)

app = pykoi.Application(debug=False, share=False)
app.add_component(chatbot_comparator)
app.run()
