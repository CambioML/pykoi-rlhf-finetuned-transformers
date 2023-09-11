FROM ubuntu:20.04
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
                   libsndfile1-dev \
                   python3.8 \
                   python3-pip \
                   python3.8-venv && \
    rm -rf /var/lib/apt/lists

# make sure to use venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
# follow the instructions here: https://cloud.google.com/tpu/docs/run-in-container#train_a_jax_model_in_a_docker_container
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --upgrade --no-cache-dir \
        pykoi

CMD ["/bin/bash"]