## running command:
## accelerate launch --num_machines 1  --num_processes 1 --mixed_precision fp16 example/rlhf/demo_rl.py

import pykoi

# use huggingface sft and reward model
config = pykoi.RLHFConfig(
    base_model_path="elinas/llama-7b-hf-transformers-4.29",  # "meta-llama/Llama-2-7b-hf",
    dataset_type="huggingface", 
    dataset_name="goldmermaid/stack_exchange_rank_10k_dataset",
    dataset_subset_rl="data",
    reward_model_path="goldmermaid/rlhf_reward_model",
    save_freq=1,
    ppo_batch_size=32,
    ppo_epochs=4,
    total_epochs=5,
    output_dir="./models/rlhf_step3_rl",
)

rlhf_step3_rl = pykoi.RLFinetuning(config)
rlhf_step3_rl.train_and_save("./models/rlhf_step3_rl")
