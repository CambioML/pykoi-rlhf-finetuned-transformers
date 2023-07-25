
# pykoi

Active learning in one unified interface :ocean:!


## Installation
To get started to `pykoi`, we recommend you test on an EC2 instance instance with the following config:
- EC2 `g5.2x` (if you want to run a pretrained model with 7B parameters)
- Deep Learning AMI GPU PyTorch 2.0.1 (Ubuntu 20.04) 20230627
- EBS: at least 100G

Once you are on your EC2 terminal, create a conda environment using:
```
conda create -n pykoi python=3.10 -y && source activate pykoi
```

Then install `pykoi` and the correlated torch version.
```
pip3 install pykoi && pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```



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
cd frontend
npm instal vite
npm run build
```