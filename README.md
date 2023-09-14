
# 🎏 pykoi: RLHF/RLAIF in one unified interface
<p align="center">
  <a href="/LICENSE"><img alt="License Apache-2.0" src="https://img.shields.io/github/license/cambioml/pykoi?style=flat-square"></a>
  <a href='http://makeapullrequest.com'><img alt='PRs Welcome' src='https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square'/></a>
</p>

[pykoi](https://www.cambioml.com/pykoi/) is an open-source python library for improving LLMs with RLHF. We provide a unified interface including RLHF/RLAIF data and feedback collection, finetuning with reinforcement learning and reward modeling, and LLM comparisons. 

## Features

`pykoi` let you easily get real-time user feedback and continuously improve your models. Here are some common applications:

### Sharable UI

Do you want to store your chat history with LLMs from OpenAI, Amazon Bedrock(:woman_technologist: building now :man_technologist:), or Huggingface? With just three lines of code, pykoi lets you to store them locally, ensuring 100% privacy. This includes launching a chatbot UI, automatically saving your chat history in your compute instance (cpu or gpu), and visualizing it on a dashboard. Explore the demos below:

- If you're using a CPU instance, check out [demo_launch_app_cpu.ipynb](https://nbviewer.org/github/CambioML/pykoi/blob/main/example/chatbot/demo_launch_app_cpu_openai.ipynb)
- If you're using a GPU instance, check out [demo_launch_app_gpu.ipynb](https://nbviewer.org/github/CambioML/pykoi/blob/main/example/chatbot/demo_launch_app_gpu_huggingface.ipynb)
- Alternatively, read our [blog](https://www.cambioml.com/docs/data_collection_feedback.html) for more information!

![Watch the video](example/image/pykoi_demo_rlaif_data_collection.gif)


### Model comparison

Comparing models is a difficult task. `pykoi` makes it easy by allowing one to directly compare the performance of multiple models to each other, with just a few lines of code. If you have multiple language models that you’d like to compare to each other on a set of prompts or via an interactive session, you can use `pk.Compare`. Check out any of the demo below: 

- If you're using a CPU instance, check out [demo_launch_app_cpu.ipynb](https://nbviewer.org/github/CambioML/pykoi/blob/main/example/chatbot/demo_model_comparator_openai.ipynb)
- If you're using a GPU instance, check out [demo_launch_app_gpu.ipynb](https://nbviewer.org/github/CambioML/pykoi/blob/main/example/chatbot/demo_model_comparator_gpu_huggingface.ipynb)
- Alternatively, read our [blog](https://www.cambioml.com/docs/model_comparison.html) for more information!
 
 ![Watch the video](example/image/pykoi_demo_model_comparison.gif)

### RLHF

Reinforcement Learning with Human Feedback (RLHF) is a unique training paradigm that blends reinforcement learning with human-in-the-loop training. The central idea is to use human evaluative feedback to refine a model's decision-making ability and guide the learning process towards desired outcomes. Researchers from [Deepmind](https://www.deepmind.com/blog/learning-through-human-feedback), [OpenAI](https://openai.com/research/learning-from-human-preferences) and [Meta Llama2](https://arxiv.org/pdf/2307.09288.pdf) have all demonstrated that RLHF is a game changer for large language models (LLMs) training.

`pykoi` allows you to easily fine-tune your model on the datasets you've collected via your `pykoi` chat or rank databases. Check our [blog](https://www.cambioml.com/docs/rlhf.html) for detailed instructions on how to use it.


## Installation
To get started with `pykoi`, you can choose to one of following compute options: CPU (e.g. your laptop) or GPU (e.g. EC2).

### Option 1: CPU (e.g. your laptop)
Installation on a CPU is simple if you have conda. If not, install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for your operating system.

First, create a conda environment on your terminal using:
```
conda create -n pykoi python=3.10 -y
conda activate pykoi
```

Then install `pykoi` and the compatible [pytorch based on your os](https://pytorch.org/get-started)
```
pip3 install pykoi
pip3 install torch 
```

### Option 2: GPU (e.g. EC2 or SageMaker)

If you are on EC2, you can launch a GPU instance with the following config:
- EC2 `g4dn.xlarge` (if you want to run a pretrained LLM with 7B parameters)
- Deep Learning AMI PyTorch GPU 2.0.1 (Ubuntu 20.04)
    <img src="example/image/readme_ec2_ami.jpg" alt="Alt text" width="50%" height="50%"/>
- EBS: at least 100G
    <img src="example/image/readme_ec2_storage.png" alt="Alt text" width="50%" height="50%"/>

Next, on your GPU instance terminal, create a conda environment using:
```
conda create -n pykoi python=3.10 -y && source activate pykoi
```

Then install `pykoi` and [pytorch based on your cuda version](https://pytorch.org/get-started).
```
pip3 install pykoi

# install torch based on cuda (e.g. cu118 means cuda 11.8)
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

Congrats you have finished the installation! 

## Dev Setup
If you are interested to contribute to us, here are the preliminary development setup.

### Backend Dev Setup
```
conda create -n pykoi python=3.10
conda activate pykoi
cd pykoi
pip3 install poetry
poetry install --no-root
```

### Frontend Dev Setup
Frontend:
```
cd pykoi/pykoi/frontend
npm install
npm run build
```
