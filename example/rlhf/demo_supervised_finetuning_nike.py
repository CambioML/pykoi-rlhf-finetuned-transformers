"""Demo for the supervised fine tuning.

python -m example.rlhf.demo_supervised_finetuning_nike
"""

from peft import LoraConfig, TaskType

from pykoi.rlhf import RLHFConfig, SupervisedFinetuning

base_model_path = "meta-llama/Llama-2-7b-chat-hf"
dataset_name = "./output_self_instructed_data_nike_10k_2023_FULL.csv"
peft_model_path = "./models/rlhf_step1_sft"
dataset_type = "local_csv"
learning_rate = 1e-3
weight_decay = 0.0
max_steps = 1600
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
log_freq = 20
eval_freq = 2000
save_freq = 200
train_test_split_ratio = 0.0001
dataset_subset_sft_train = 999999999
size_valid_set = 0

r = 8
lora_alpha = 16
lora_dropout = 0.05
bias = "none"
task_type = TaskType.CAUSAL_LM

lora_config = LoraConfig(
    r=r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias=bias,
    task_type=task_type,
)


# run supervised finetuning
config = RLHFConfig(
    base_model_path=base_model_path,
    dataset_type=dataset_type,
    dataset_name=dataset_name,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    max_steps=max_steps,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    log_freq=log_freq,
    eval_freq=eval_freq,
    save_freq=save_freq,
    train_test_split_ratio=train_test_split_ratio,
    dataset_subset_sft_train=dataset_subset_sft_train,
    size_valid_set=size_valid_set,
    lora_config_rl=lora_config,
)
rlhf_step1_sft = SupervisedFinetuning(config)
rlhf_step1_sft.train_and_save(peft_model_path)
