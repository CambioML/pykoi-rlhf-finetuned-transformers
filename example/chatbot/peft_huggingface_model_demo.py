"""Demo for the chatbot application."""
import plotano.cambio as cb

###################################################################################
# Creating a Huggingface model tiiuae/falcon-7b (EC2 g5.4xlarge with 100GB space) #
###################################################################################
# only run this after run rlfh/sft_demo.py to fine tune the model

model = cb.ModelFactory.create_model(
    model_name="peft_huggingface",
    base_model_path="meta-llama/Llama-2-7b-hf",
    lora_model_path="/home/ubuntu/plotano/models/rlhf_step1_sft",
)

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
