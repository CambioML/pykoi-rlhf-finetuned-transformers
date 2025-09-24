
# 🎏 pykoi: RLHF/RLAIF in one unified interface
<p align="center">
  <a href="/LICENSE"><img alt="License Apache-2.0" src="https://img.shields.io/github/license/cambioml/pykoi?style=flat-square"></a>
  <a href="https://pypi.org/project/pykoi"><img src="https://img.shields.io/pypi/v/pykoi.svg" alt="pypi_status" /></a>
  <a href="https://github.com/cambioml/pykoi/graphs/commit-activity"><img alt="Commit activity" src="https://img.shields.io/github/commit-activity/m/cambioml/pykoi?style=flat-square"/></a>
  <a href="https://join.slack.com/t/cambiomlworkspace/shared_invite/zt-1zes33rmt-20Rag043uvExUaUdvt5_xQ"><img src="https://badgen.net/badge/Join/Community/cyan?icon=slack" alt="Slack" /></a>
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

### RAG
Implement a Retrieval-Augmented Generation (RAG) Chatbot quickly with `pykoi`. Upload your own documents to create context-aware responses on top of a pretrained LLM, such as from OpenAI or Huggingface.

![Watch the video](example/image/pykoi_demo_rag_chatbot_upload.gif)

Then chat with the RAG Chatbot, where you can select specific sources for the text generation, as well as see which sources the model used to generate the response.

![Watch the video](example/image/pykoi_demo_rag_chatbot_chat.gif)

Finally, you can save a modified response to help improve the model. This is a great way to collect data for RLHF.

![Watch the video](example/image/pykoi_demo_rag_chatbot_modify.gif)


## Installation
To get started with pykoi, you can choose from any of the installation options. The choice should be based on the features you need (e.g., RAG, RLHF or all) and the compute resources you have, such as a CPU (e.g., your laptop) or GPU (e.g., AWS EC2 or SageMaker).

### Option 1: RAG (CPU)
This option allows you to run RAG on a CPU using either the OpenAI API or the Anthropic Claude2 API. Installation of RAG (CPU) is simple if you have conda. If not, install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for your operating system.

First, create a conda environment on your terminal using:
```
conda create -n pykoi python=3.10 -y
conda activate pykoi  # some OS requires `source activate pykoi`
```

Then install `pykoi` and the compatible [pytorch based on your os](https://pytorch.org/get-started/locally/)
```
pip3 install "pykoi[rag]"
pip3 install torch
```

### Option 2: RAG (GPU)
This option allows you to run RAG on a GPU using an open-source LLM from HuggingFace. Here's a quick [tutorial](#ec2-dev-setup) on setting up an EC2 GPU instance for the installation below.

On your GPU instance terminal, create a conda environment using:
```
conda create -n pykoi python=3.10 -y && source activate pykoi
```

Then install `pykoi` and [pytorch based on your cuda version](https://pytorch.org/get-started/locally/). You can find your CUDA version via `nvcc -V`.
```
pip3 install "pykoi[rag, huggingface]"

pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121  # cu121 means cuda 12.1
```

### Option 3: RLHF (GPU)
This option allows you to train LLM via RLHF on a GPU. Here's a quick [tutorial](#ec2-dev-setup) on setting up an EC2 GPU instance for the installation below.

On your GPU instance terminal, create a conda environment using:
```
conda create -n pykoi python=3.10 -y && source activate pykoi
```

Then install `pykoi` and [pytorch based on your cuda version](https://pytorch.org/get-started/locally/). You can find your CUDA version via `nvcc -V`.
```
pip3 install "pykoi[rlhf]"

pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121  # cu121 means cuda 12.1
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
```

Then, based the feature you need to develop, run one or more installation options below. We recommend install all the options below although it may take ~3 minutes longer.

- Option 1: RAG (CPU)
  ```
  poetry install --no-root --extras rag
  ```
- Option 2: RAG (GPU)
  ```
  poetry install --no-root --extras "rag huggingface"
  ```
- Option 3: RLHF (GPU)
  ```
  poetry install --no-root --extras rlhf
  ```

Finally, if you are on a GPU, install [pytorch based on your cuda version](https://pytorch.org/get-started/locally/). You can find your CUDA version via `nvcc -V`.
```
pip3 uninstall -y torch
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121  # cu121 means cuda 12.1
```

Now you are done with installing, try to modify one of the examples and run the below command!

```
python -m example.retrieval_qa.retrieval_qa_huggingface_demo
```

### Frontend Dev Setup
```
cd pykoi/pykoi/frontend
npm install
npm run build
```

### EC2 Dev Setup
If you are on EC2, you can launch a GPU instance with the following config:
- EC2 `g5.2xlarge` (if you want to run a pretrained LLM with 7B parameters)
- Deep Learning AMI PyTorch GPU 2.0.1 (Ubuntu 20.04)
    <img src="example/image/readme_ec2_ami.jpg" alt="Alt text" width="75%" height="75%"/>
- EBS: at least 100G

    <img src="example/image/readme_ec2_storage.png" alt="Alt text" width="50%" height="50%"/>

## Related Resources
- [Energent.ai](https://energent.ai) – primer on automating data workflows with CambioML.
      
### 🌐 Learn more about Energent.ai
If you’re curious about what is being built at Energent.ai and how these tools fit into the bigger picture, here are a few links worth checking out:
  - [**Main website / Hompage**](https://www.energent.ai)  
  A quick overview of what Energent is all about — building AI agents that automate real-world tasks, especially in messy, complex workflows.

- [**Energent.ai app (live demo)**](https://app.energent.ai)  
  Live demo environment where you can create, test, and run your own agents. Hands-on version of the platform - This is where users can build and test agents directly in the browser.

- [**Gallery of some app use cases**](https://app.energent.ai/gallery)  
  A collection of real examples and workflows built on Energentai platform. Helpful if you want to see what users are building with it, and useful for understanding what kinds of tasks the platform is good at.

- [**About the team**](https://www.energent.ai/en/company)  
  A bit of background on the people behind Energent and what they're aiming to build long-term.

- [**Product blog**](https://www.energent.ai/en/blog)  
 Occasional deep dives on product updates, dev notes, new feature writeups and design decisions. Good place to get a feel for the thinking behind the tools.
