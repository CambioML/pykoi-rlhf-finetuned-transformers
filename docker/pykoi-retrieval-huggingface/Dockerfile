# Use a Python base image
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app/

# Install necessary system packages and clean up
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy only necessary project files into the container
COPY pyproject.toml poetry.lock /app/

# Install project dependencies using Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --extras "rag huggingface" \
    && pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121 \
    && pip uninstall -y torchvision \
    && rm -rf /root/.cache/

# Copy the project files into the container
COPY example /app/example
COPY pykoi /app/pykoi

# Set the necessary environment variables to enable GPU
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV RETRIEVAL_MODEL=databricks/dolly-v2-3b

# Set entrypoint to run your command
CMD ["python", "-u", "-m", "example.retrieval_qa.retrieval_qa_huggingface_demo"]

