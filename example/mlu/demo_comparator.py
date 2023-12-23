"""Demo for the chatbot application using multiple model endpoint."""

from transformers import AutoModelForCausalLM, AutoTokenizer

from pykoi import Application
from pykoi.chat.llm.huggingface import HuggingfaceModel
from pykoi.component import Compare

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

#################################
# Creating a chatbot comparator #
#################################
# pass in a list of models to compare
model_name = ["falcon-rw-1b", "falcon-7b", "dolly-v2-3b"]
models = [hf_model_1, hf_model_2, hf_model_3]
tokenizers = [hf_tokenizer_1, hf_tokenizer_2, hf_tokenizer_3]

models_list = [
    HuggingfaceModel.create(model=model, tokenizer=tokenizer, name=name, max_length=100)
    for model, tokenizer, name in zip(models, tokenizers, model_name)
]

chatbot_comparator = Compare(models=models_list)
app = Application()
app.add_component(chatbot_comparator)
app.run()
