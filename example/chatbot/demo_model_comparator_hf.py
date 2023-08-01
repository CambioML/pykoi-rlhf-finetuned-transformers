"""Demo for the chatbot application using multiple model endpoint."""
import pykoi

###################################################################################
# Creating a Huggingface model tiiuae/falcon-rw-1b (EC2 g4.2xlarge with 100GB space) #
###################################################################################
huggingface_model_1 = pykoi.ModelFactory.create_model(
    model_source="huggingface",
    pretrained_model_name_or_path="tiiuae/falcon-rw-1b",
)

###################################################################################
# Creating a Huggingface model tiiuae/falcon-7b (EC2 g4.2xlarge with 100GB space) #
###################################################################################
huggingface_model_2 = pykoi.ModelFactory.create_model(
    model_source="huggingface",
    pretrained_model_name_or_path="tiiuae/falcon-7b",
)

###################################################################################
# Creating a Huggingface model databricks/dolly-v2-3b (EC2 g4.2xlarge with 100GB space) #
###################################################################################
huggingface_model_3 = pykoi.ModelFactory.create_model(
    model_source="huggingface",
    pretrained_model_name_or_path="databricks/dolly-v2-3b",
)

###################################################################################
# Creating a Huggingface model meta-llama/Llama-2-7b-hf (EC2 g4.2xlarge with 100GB space) #
###################################################################################
# only run this after run rlfh/sft_demo.py to fine tune the model
# peft_huggingface_model = pykoi.ModelFactory.create_model(
#     model_source="peft_huggingface",
#     base_model_path="meta-llama/Llama-2-7b-hf",
#     lora_model_path="/home/ubuntu/pykoi/models/rlhf_step1_sft",
# )
# questions = [...]
#################################
# Creating a chatbot comparator #
#################################
# pass in a list of models to compare
chatbot_comparator = pykoi.Compare(models=[huggingface_model_1, huggingface_model_2])
# or add models later
chatbot_comparator.add(huggingface_model_3)

app = pykoi.Application(debug=False, share=True, username="rachel", password="rachel")
app.add_component(chatbot_comparator)
app.run()
