FROM python:3.10
LABEL maintainer="Pykio Team"
LABEL repository="Pykoi"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   git-lfs \
                   curl \
                   ca-certificates \
                   libsndfile1-dev

# make sure to use venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
# follow the instructions here: https://cloud.google.com/tpu/docs/run-in-container#train_a_jax_model_in_a_docker_container
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --upgrade --no-cache-dir \
    pykoi

CMD ["/bin/bash"]