"""
Demo for the chatbot application using multiple open source LLMs from Huggingface.

- Prerequisites:
    To run this jupyter notebook, you need a `pykoi` environment with the `huggingface` option. 
    You can follow [the installation guide](https://github.com/CambioML/pykoi/tree/install#option-2-rag-gpu) 
    to set up the environment. 
- Run the demo:
    1. On terminal and `~/pykoi` directory, run
        ```
        python -m example.chatbot.demo_model_comparator_gpu_huggingface
        ```
"""
from pykoi import Application
from pykoi.chat import ModelFactory
from pykoi.component import Compare


###################################################################################
# Creating a Huggingface model tiiuae/falcon-rw-1b (EC2 g4.2xlarge with 100GB space) #
###################################################################################
huggingface_model_1 = ModelFactory.create_model(
    model_source="huggingface",
    pretrained_model_name_or_path="tiiuae/falcon-rw-1b",
)

###################################################################################
# Creating a Huggingface model tiiuae/falcon-7b (EC2 g4.2xlarge with 100GB space) #
###################################################################################
huggingface_model_2 = ModelFactory.create_model(
    model_source="huggingface",
    pretrained_model_name_or_path="tiiuae/falcon-7b",
)

###################################################################################
# Creating a Huggingface model databricks/dolly-v2-3b (EC2 g4.2xlarge with 100GB space) #
###################################################################################
huggingface_model_3 = ModelFactory.create_model(
    model_source="huggingface",
    pretrained_model_name_or_path="databricks/dolly-v2-3b",
)

###################################################################################
# Creating a Huggingface model elinas/llama-7b-hf-transformers-4.29 (EC2 g4.2xlarge with 100GB space) #
###################################################################################
# only run this after run rlfh/sft_demo.py to fine tune the model
# peft_huggingface_model = pykoi.ModelFactory.create_model(
#     model_source="peft_huggingface",
#     base_model_path="elinas/llama-7b-hf-transformers-4.29",
#     lora_model_path="/home/ubuntu/pykoi/models/rlhf_step1_sft",
# )
# questions = [...]
#################################
# Creating a chatbot comparator #
#################################
# pass in a list of models to compare
chatbot_comparator = Compare(models=[huggingface_model_1, huggingface_model_2])
# or add models later
chatbot_comparator.add(huggingface_model_3)

app = Application(debug=False, share=True, username="rachel", password="rachel")
app.add_component(chatbot_comparator)
app.run()
