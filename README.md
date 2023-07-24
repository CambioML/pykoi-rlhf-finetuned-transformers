
# pykoi

## Setup
```
conda create -n pykoi python=3.10
conda activate pykoi
cd pykoi
pip3 install poetry
poetry install --no-root
```
## Run
```
python -m example.chatbot.demo
```

## Development
Frontend:
```
cd frontend
npm run build
```

Backend:
```
python -m example.chatbot.openai_model_demo
```
