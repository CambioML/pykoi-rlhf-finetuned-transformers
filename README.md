
# pykoi

## User Setup
```
pip3 install pykoi

# Assume you are running on EC2 with Deep Learning AMI GPU PyTorch 2.0.1 (Ubuntu 20.04) 20230627
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```



## Backend Dev Setup
```
conda create -n pykoi python=3.10
conda activate pykoi
cd pykoi
pip3 install poetry
poetry install --no-root
```

## Frontend Dev Setup
Frontend:
```
cd frontend
npm instal vite
npm run build
```