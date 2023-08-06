"""Demo for the chatbot application."""
import pykoi

###################################################################################
# Creating a Huggingface model tiiuae/falcon-7b (EC2 g5.4xlarge with 100GB space) #
###################################################################################
# only run this after run rlfh/sft_demo.py to fine tune the model

model = pykoi.ModelFactory.create_model(
    model_source="peft_huggingface",
    base_model_path="elinas/llama-7b-hf-transformers-4.29",
    lora_model_path="/home/ubuntu/pykoi/models/rlhf_step1_sft",
)

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
app = pykoi.Application(debug=False, share=False)
app.add_component(chatbot)
app.add_component(dashboard)
app.run()
