# Examples
This directory contains examples of how to use the library.

## 1. Use with OpenAI
### 1.1 setup local machine
#### 1.1.1 Install dependencies
```
pip3 install pykoi

# install other dependencies
pip3 install openai
```

#### 1.1.2 Run
```
python openai_model_demo.py
```

## 2. Use with HuggingFace Model
### 2.1 setup AWS EC2 Ubuntu Deep Learning AMI GPU PyTorch 2.0.1 (Ubuntu 20.04) 20230627 AMI g5.4xlarge with 100GB space
#### 2.1.1 Install dependencies
```
pip3 install pykoi

# install torch. update accordingly based on your os cuda version at https://pytorch.org/get-started
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

#### 2.1.2 Run
```
python huggingface_model_demo.py
```
