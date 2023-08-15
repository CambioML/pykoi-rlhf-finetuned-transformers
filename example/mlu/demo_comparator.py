"""Demo for the chatbot application using multiple model endpoint."""

from transformers import AutoModelForCausalLM, AutoTokenizer

import pykoi


######################################################################################
# Creating a Huggingface model tiiuae/falcon-rw-1b (EC2 g4.2xlarge with 100GB space) #
######################################################################################
print("create model 1...")
hf_model_1 = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="tiiuae/falcon-rw-1b",
    trust_remote_code=True,
    load_in_8bit=True,
    device_map="auto",
)

print("create tokenizer 1...")
hf_tokenizer_1 = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="tiiuae/falcon-rw-1b",
    trust_remote_code=True,
    load_in_8bit=True,
    device_map="auto",
)

model_1 = pykoi.chat.llm.huggingface.HuggingfaceModel.create(
    model=hf_model_1,
    tokenizer=hf_tokenizer_1,
    name="falcon-rw-1b",
    max_length=100,
)

###################################################################################
# Creating a Huggingface model tiiuae/falcon-7b (EC2 g4.2xlarge with 100GB space) #
###################################################################################
print("create model 2...")
hf_model_2 = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="tiiuae/falcon-7b",
    trust_remote_code=True,
    load_in_8bit=True,
    device_map="auto",
)

print("create tokenizer 2...")
hf_tokenizer_2 = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="tiiuae/falcon-7b",
    trust_remote_code=True,
    load_in_8bit=True,
    device_map="auto",
)

model_2 = pykoi.chat.llm.huggingface.HuggingfaceModel.create(
    model=hf_model_2,
    tokenizer=hf_tokenizer_2,
    name="falcon-7b",
    max_length=100,
)

#########################################################################################
# Creating a Huggingface model databricks/dolly-v2-3b (EC2 g4.2xlarge with 100GB space) #
#########################################################################################
print("create model 3...")
hf_model_3 = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="databricks/dolly-v2-3b",
    trust_remote_code=True,
    load_in_8bit=True,
    device_map="auto",
)

print("create tokenizer 3...")
hf_tokenizer_3 = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="databricks/dolly-v2-3b",
    trust_remote_code=True,
    load_in_8bit=True,
    device_map="auto",
)

model_3 = pykoi.chat.llm.huggingface.HuggingfaceModel.create(
    model=hf_model_3,
    tokenizer=hf_tokenizer_3,
    name="dolly-v2-3b",
    max_length=100,
)

#################################
# Creating a chatbot comparator #
#################################
# pass in a list of models to compare
chatbot_comparator = pykoi.Compare(models=[model_1, model_2])
# or add models later
chatbot_comparator.add(model_3)

app = pykoi.Application()
app.add_component(chatbot_comparator)
app.run()
