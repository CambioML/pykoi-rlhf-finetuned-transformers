# Overview
In this folder, we create the different dockerfiles for using pykoi.

1. `pykoi-cpu`: The base image for the cpu-based usage.
2. `pykoi-cpu-custom`: When you run this docker image, try to modify the `app.py` and mount it when running the docker container.

To run a docker container, we can use the following command:
```bash
docker run -dp 5000:5000 -v $(pwd)/app.py:/app/app.py \
        --name alex_test \
        weialexchen/pykoi-cpu:app
```