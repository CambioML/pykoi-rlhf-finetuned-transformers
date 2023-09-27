"""
Demo for the chatbot application using multiple OpenAI models.

- Prerequisites:
    To run this jupyter notebook, you need a `pykoi` environment with the `rag` option.
    You can follow [the installation guide](https://github.com/CambioML/pykoi/tree/install#option-1-rag-cpu)
    to set up the environment.
- Run the demo:
    1. Create an `.env` file in the `~/pykoi/` directory. 
    2. Replace `your_api_key` with your OpenAI API key in the `.env` file, e.g.
        ```
        OPENAI_API_KEY=your_api_key
        ```
    3. On terminal and `~/pykoi` directory, run
        ```
        python -m example.comparator.demo_model_comparator_cpu_openai
        ```
"""

import os

from dotenv import load_dotenv

from pykoi import Application
from pykoi.chat import ModelFactory
from pykoi.component import Compare


##########################################################
# Creating an OpenAI model (requires an OpenAI API key) #
##########################################################
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Creating an OpenAI model
openai_model_1 = ModelFactory.create_model(
    model_source="openai", name="openai_babbage", api_key=api_key, engine="babbage"
)
openai_model_2 = ModelFactory.create_model(
    model_source="openai", name="openai_curie", api_key=api_key, engine="curie"
)
openai_model_3 = ModelFactory.create_model(
    model_source="openai", name="openai_davinci", api_key=api_key, engine="davinci"
)

###################################################################################
# Creating a Huggingface model tiiuae/falcon-7b (EC2 g5.2xlarge with 100GB space) #
###################################################################################
# huggingface_model = pykoi.ModelFactory.create_model(
#     model_source="huggingface",
#     pretrained_model_name_or_path="tiiuae/falcon-7b",
# )

###################################################################################
# Creating a Huggingface model tiiuae/falcon-7b (EC2 g5.2xlarge with 100GB space) #
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
chatbot_comparator = Compare(models=[openai_model_1, openai_model_2])
chatbot_comparator.add(openai_model_3)
# or add models later
# chatbot_comparator.add(huggingface_model)
# chatbot_comparator.add(peft_huggingface_model)

app = Application(debug=False, share=False)
app.add_component(chatbot_comparator)
app.run()
