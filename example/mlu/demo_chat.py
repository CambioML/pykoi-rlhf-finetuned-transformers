"""Demo for the MLU chatbot application."""

from transformers import AutoModelForCausalLM, AutoTokenizer

import pykoi


###################################################################################
# Creating a Huggingface model tiiuae/falcon-7b (EC2 g5.4xlarge with 100GB space) #
###################################################################################
print("create model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="tiiuae/falcon-7b",
    trust_remote_code=True,
    load_in_8bit=True,
    device_map="auto",
)

print("create tokenizer...")
hf_tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="tiiuae/falcon-7b",
    trust_remote_code=True,
    load_in_8bit=True,
    device_map="auto",
)

print("create pykoi model component for UI...")
model = pykoi.chat.llm.huggingface.HuggingfaceModel.create(
    model=hf_model,
    tokenizer=hf_tokenizer,
    name="falcon-7b",
    max_length=100,
)

#####################################
# Creating a chatbot with the model #
#####################################
database = pykoi.QuestionAnswerDatabase(debug=True)
chatbot = pykoi.Chatbot(model=model, feedback="vote")
dashboard = pykoi.Dashboard(database=database)

###########################################################
# Starting the application and add chatbot as a component #
###########################################################
# Create the application
app = pykoi.Application()
app.add_component(chatbot)
app.add_component(dashboard)
app.run()
