# Examples
This directory contains examples of how to use the library.

## 1. Use with OpenAI
### 1.1 setup local machine
#### 1.1.1 Install dependencies
```
pip3 install plotano

# install other dependencies
pip3 install openai
```
#### 1.1.2 Edit
uncomment the following lines in chatbot_demo.py and put your api_key in the variable
```
# # enter openai api key here
# api_key = ""

# # Creating an OpenAI model
# model = cb.ModelFactory.create_model(
#     model_name="openai",
#     api_key=api_key)
```

#### 1.1.3 Run
```
python demo.py
```

## 2. Use with HuggingFace Model
### 2.1 setup AWS EC2 Ubuntu Deep Learning AMI GPU PyTorch 2.0.1 (Ubuntu 20.04) 20230627 AMI g5.4xlarge with 100GB space
#### 2.1.2 Install dependencies
```
pip3 install plotano

# install torch. update accordingly based on your os cuda version at https://pytorch.org/get-started
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
#### 2.1.2 Edit
uncomment the following lines in chatbot_demo.py and put your api_key in the variable
```
# model = cb.ModelFactory.create_model(
#     model_name="huggingface",
#     pretrained_model_name_or_path="tiiuae/falcon-7b",
#     trust_remote_code=True,
#     load_in_8bit=True)
```

#### 2.1.3 Run
```
python demo.py
```