"""
Demo for the chatbot application using open source LLMs from Huggingface Peft.

- Prerequisites:
    To run this jupyter notebook, you need a `pykoi` environment with the `huggingface` option. 
    You can follow [the installation guide](https://github.com/CambioML/pykoi/tree/install#option-2-rag-gpu) 
    to set up the environment. 
- Run the demo:
    1. On terminal and `~/pykoi` directory, run
        ```
        python -m example.chatbot.demo_launch_app_gpu_huggingface_peft
        ```
"""

from pykoi import Application
from pykoi.chat import ModelFactory, QuestionAnswerDatabase
from pykoi.component import Chatbot, Dashboard

###################################################################################
# Creating a Huggingface model tiiuae/falcon-7b (EC2 g5.4xlarge with 100GB space) #
###################################################################################
# only run this after run rlfh/sft_demo.py to fine tune the model

model = ModelFactory.create_model(
    model_source="peft_huggingface",
    base_model_path="meta-llama/Llama-2-7b-chat-hf",
    lora_model_path="/home/ubuntu/pykoi/models/rlhf_step1_sft",
)

#####################################
# Creating a chatbot with the model #
#####################################
database = QuestionAnswerDatabase(debug=True)
chatbot = Chatbot(model=model, feedback="vote")
# chatbot = pykoi.Chatbot(model=model, feedback="rank")
dashboard = Dashboard(database=database)

###########################################################
# Starting the application and add chatbot as a component #
###########################################################
# Create the application
# app = pykoi.Application(debug=False, share=True)
app = Application(debug=False, share=False)
app.add_component(chatbot)
app.add_component(dashboard)
app.run()
