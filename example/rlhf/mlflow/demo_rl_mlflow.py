"""
huggingface-cli login --token $HUGGINGFACE_TOKEN

accelerate config

LOCAL_DIR=/home/ubuntu/pykoi # change this to your local path

export PYTHONPATH=$PYTHONPATH:${LOCAL_DIR}

accelerate launch --num_machines 1  --num_processes 1 --mixed_precision fp16 ${LOCAL_DIR}/example/rlhf/mlflow/demo_rl_mlflow.py
"""
# accelerate launch --num_machines 1  --num_processes 1 --mixed_precision fp16 example/rlhf/mlflow/demo_rl_mlflow.py

from pykoi.rlhf import RLHFConfig
from pykoi.rlhf import RLFinetuning
import mlflow
import datetime

# Log into huggingface with token if it is not done so in the command line.
# https://huggingface.co/docs/huggingface_hub/quick-start#login
# https://huggingface.co/settings/tokens

# from huggingface_hub import login
# login(token="")

# Set up mlflow experiment name.
mlflow.set_tracking_uri("example/rlhf/mlflow/mlruns")
experiment = "rlhf_step3_rl"
current_time = str(datetime.datetime.now())
mlflow_experiment_name = '/'.join([experiment, current_time])
mlflow.set_experiment(mlflow_experiment_name)

# Set pykoi parameters.
base_model_path = "example/rlhf/mlflow/models/rlhf_step1_sft"
dataset_type = "local_db"
reward_model_path = "example/rlhf/mlflow/models/rlhf_step2_rw"
dataset_subset_rl = "data"
save_freq = 1
ppo_batch_size = 32
ppo_epochs = 4
total_epochs = 5
output_dir = "example/rlhf/mlflow/models/rlhf_step3_rl"

# Manually log pykoi parameters into mlflow. Torch level parameters are automatically logged.
mlflow.log_param("pykoi_base_model_path", base_model_path)
mlflow.log_param("pykoi_dataset_type", dataset_type)
mlflow.log_param("pykoi_reward_model_path", reward_model_path)
mlflow.log_param("pykoi_dataset_subset_rl", dataset_subset_rl)
mlflow.log_param("pykoi_save_freq", save_freq)
mlflow.log_param("pykoi_ppo_batch_size", ppo_batch_size)
mlflow.log_param("pykoi_ppo_epochs", ppo_epochs)
mlflow.log_param("pykoi_total_epochs", total_epochs)
mlflow.log_param("pykoi_output_dir", output_dir)

# Use huggingface sft and reward model
# Training metrics are automatically logged into mlflow.
config = RLHFConfig(
    base_model_path=base_model_path,    #"elinas/llama-7b-hf-transformers-4.29", 
    dataset_type=dataset_type,
    # dataset_type="huggingface", 
    # dataset_name="goldmermaid/stack_exchange_rank_10k_dataset",
    dataset_subset_rl=dataset_subset_rl,
    reward_model_path=reward_model_path, #"cambioml/rlhf_reward_model",
    save_freq=save_freq,
    ppo_batch_size=ppo_batch_size,
    ppo_epochs=ppo_epochs,
    total_epochs=total_epochs,
    output_dir=output_dir,
)

rlhf_step3_rl = RLFinetuning(config)
rlhf_step3_rl.train_and_save(output_dir)

# Save the trained model into mlflow artifacts.
mlflow.log_artifacts(output_dir)