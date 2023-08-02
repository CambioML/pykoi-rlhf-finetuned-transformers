## pip install xformers ## TODO
## running command:
## accelerate launch --num_machines 1  --num_processes 1 --mixed_precision fp16 example/rlhf/demo_rl.py

import pykoi

# use huggingface sft and reward model
# config = pykoi.RLHFConfig(
#     base_model_path="elinas/llama-7b-hf-transformers-4.29",  # "meta-llama/Llama-2-7b-hf",
#     dataset_type="huggingface",  ## "local_db",
#     dataset_name="goldmermaid/stack_exchange_rank_10k_dataset",  ##"lvwerra/stack-exchange-paired", ## "/home/ubuntu/git/pykoi/example/notebook/qd.db",
#     dataset_subset_rl="data",
#     reward_model_path="goldmermaid/rlhf_reward_model",
#     save_freq=1,
#     ppo_batch_size=32,
#     ppo_epochs=1,
#     total_epochs=1,
#     output_dir="./models/rlhf_step3_rl",
# )

# bring your own step1 sft and step 2 reward model
config = pykoi.RLHFConfig(
    step1_sft_model_path="./models/rlhf_step1_sft",  # "meta-llama/Llama-2-7b-hf",
    step2_reward_model_path="./models/rlhf_step2_rw",
    output_dir="./models/rlhf_step3_rl",
    dataset_type="local_db",
    save_freq=1,
    ppo_batch_size=32,
    ppo_epochs=1,
    total_epochs=1,
)

rlhf_step3_rl = pykoi.RLFinetuning(config)
rlhf_step3_rl.train_and_save("./models/rlhf_step3_rl")
