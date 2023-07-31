"""RLHF configuration file."""

from dataclasses import dataclass, field
from typing import (
    List,
    Optional
)

from accelerate import Accelerator
from peft import LoraConfig, TaskType


@dataclass
class RLHFConfig:
    """
    This file contains the configuration parameters for the RLHF (Reinforcement Learning for Humans Feedback) model.
    The parameters are divided into three steps:
        - Step 1: SFT (Supervised Fine-Tuning) parameters
        - Step 2: Reward Modeling parameters
        - Step 3: Reinforcement Learning parameters
    """

    base_model_path: str = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "Huggingface model name or a local path to the base model."},
    )
    dataset_type: Optional[str] = field(
        default="local_db",
        metadata={"help": "'local_db':load from your local database `qd.db` path; \
                  'local_csv':load from a local csv path; 'huggingface': load a huggingface dataset."})
    dataset_name: Optional[str] = field(
        default="qd.db",
        metadata={"help": "A local path to a csv dataset or a database; Or a Huggingface "
                  "dataset name (e.g. 'lvwerra/stack-exchange-paired')."})
    train_test_split_ratio: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The ratio represents the proportion of the test dataset to \
                  include in the train and test split."
        },
    )
    streaming: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use streaming."}
    )
    shuffle_buffer: Optional[int] = field(
        default=5000, metadata={"help": "Size of the shuffle buffer."}
    )
    max_seq_length: Optional[int] = field(
        default=512, metadata={"help": "Maximum sequence length."}
    )
    evaluation_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": "The evaluation strategy to adopt during training."},
    )
    # batch_size: int = field(
    #     default=8,
    #     metadata={"help": "Batch size."})
    per_device_train_batch_size: Optional[int] = field(
        default=2, metadata={"help": "Batch size per device for training."}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=8, metadata={"help": "Batch size per device for evaluation."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "Number of steps for gradient accumulation."}
    )
    eos_token_id: Optional[int] = field(
        default=49152, metadata={"help": "End-of-sequence token ID."}
    )
    learning_rate: Optional[float] = field(
        default=1e-5, metadata={"help": "Learning rate."}
    )
    weight_decay: Optional[float] = field(
        default=0.01, metadata={"help": "Weight decay."}
    )
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu."}
    )
    fp16: Optional[bool] = field(default=True, metadata={"help": "Enable FP16."})
    bf16: Optional[bool] = field(default=False, metadata={"help": "Enable BF16."})
    load_in_8bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether load the model weights in 8-bit or not."},
    )
    device_map: Optional[dict] = field(
        default_factory=lambda: {"": Accelerator().process_index},
        metadata={
            "help": "specify the mapping of model layers to specific devices, such as different GPUs \
                  in a multi-GPU setup. This can be helpful for distributing the computational load of a \
                  large model across multiple GPUs."
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Enable gradient checkpointing."}
    )
    seed: Optional[int] = field(default=0, metadata={"help": "Random seed."})
    num_workers: Optional[int] = field(
        default=None, metadata={"help": "Number of workers."}
    )
    output_dir: Optional[str] = field(
        default="./rlhf_checkpoints",
        metadata={"help": "Output directory for all model weights."},
    )
    log_freq: Optional[int] = field(default=1, metadata={"help": "Logging frequency."})
    eval_freq: Optional[int] = field(
        default=1000, metadata={"help": "Evaluation frequency."}
    )
    save_freq: Optional[int] = field(
        default=1000, metadata={"help": "Model saving frequency."}
    )
    push_to_hub: Optional[bool] = field(
        default=False, metadata={"help": "Whether push to Huggingface Hub or not."}
    )

    ## Step 1 SFT parameters
    max_steps: Optional[int] = field(
        default=5, metadata={"help": "Maximum number of training steps."}
    )
    dataset_subset_sft: Optional[str] = field(
        default="data/finetune",
        metadata={"help": "Subset folder of the dataset to use."},
    )
    dataset_subset_sft_train: Optional[int] = field(
        default=10000,
        metadata={"help": "The size of the subset of the training data to use."},
    )
    split: Optional[str] = field(
        default="train", metadata={"help": "Dataset split to use."}
    )
    question_title: Optional[str] = field(
        default="Question",
        metadata={"help": "the column name of questions from the database."},
    )
    answer_title: Optional[str] = field(
        default="Answer",
        metadata={"help": "the column name of answers from the database."},
    )
    size_valid_set: Optional[int] = field(
        default=4000, metadata={"help": "Size of the validation/eval set."}
    )
    sft_lora_path: Optional[str] = field(
        default="step1_supervised_finetuning_lora_final/",
        metadata={
            "help": "Output directory for step 1 supervised finetuning's Lora weights."
        },
    )
    sft_merged_path: Optional[str] = field(
        default="step1_supervised_finetuning_merged/",
        metadata={
            "help": "Output directory for step 1 supervised finetuning's merged weights."
        },
    )
    lr_scheduler_type_sft: Optional[str] = field(
        default="cosine", metadata={"help": "Type of learning rate scheduler."}
    )
    num_warmup_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of warmup steps for the scheduler."}
    )
    lora_config_rl: Optional[LoraConfig] = field(
        default=LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
        metadata={"help": "LoRA configuration."},
    )

    # Step 2 reward modeling parameters
    reward_model_path: Optional[str] = field(
        default="databricks/dolly-v2-3b",
        metadata={
            "help": "Huggingface model name or a local path to the reward model."
        },
    )
    reward_lora_path: Optional[str] = field(
        default="step1_supervised_finetuning_lora_final/",
        metadata={
            "help": "Output directory for step 1 supervised finetuning's Lora weights."
        },
    )
    reward_merged_path: Optional[str] = field(
        default="step1_supervised_finetuning_merged/",
        metadata={
            "help": "Output directory for step 1 supervised finetuning's merged weights."
        },
    )
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    dataset_reward_folder: Optional[str] = field(
        default="data/reward",
        metadata={"help": "Subset folder of the reward dataset to use."},
    )
    dataset_eval_folder: Optional[str] = field(
        default="data/evaluation",
        metadata={"help": "Subset folder of the evaluation dataset to use."},
    )
    reward_num_of_data: Optional[int] = field(
        default=1000,
        metadata={"help": "The size of the subset of the training data to use."},
    )
    max_seq_length_reward: Optional[int] = field(
        default=512, metadata={"help": "Maximum sequence length."}
    )
    # dataset_subset_reward_eval: Optional[int] = field(
    #     default=400,
    #     metadata={"help": "The size of the subset of the validation/eval data to use."})
    reward_epochs: Optional[int] = field(
        default=10,
        metadata={"help": "The number of training epochs for reward modeling."},
    )
    deepspeed: Optional[str] = field(
        default=None,  # TODO
        metadata={"help": "Path to deepspeed config if using deepspeed."},
    )
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to remove unused columns from the dataset."},
    )
    label_names: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "List of column names in the dataset to use as labels."},
    )
    logging_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": "The strategy used for logging during training."},
    )
    logging_steps: Optional[int] = field(
        default=10, metadata={"help": "The number of steps between each logging."}
    )
    # callbacks: Optional[List[TrainerCallback]] = field(
    #     default=[], ## PeftSavingCallback()
    #     metadata={"help": "The callbacks to use for training."}),
    # optim: Optional[str] = field(
    #     default="adamw_hf", metadata={"help": "The optimizer to use."})
    # lr_scheduler_type_rw: str = field(
    #     default="linear",
    #     metadata={"help": "Type of learning rate scheduler."})
    lora_config_reward: Optional[LoraConfig] = field(
        default=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
        ),
        metadata={"help": "LoRA configuration."},
    )

    # Step 3 RL parameters
    dataset_subset_rl: Optional[str] = field(
        default="data",
        metadata={"help": "Subset folder of the dataset to use."}, ## TODO
    )
    dataset_subset_rl_train: Optional[int] = field(
        default=10000,
        metadata={"help": "The size of the subset of the training data to use."},
    )
    adafactor: Optional[bool] = field(
        default=False, metadata={"help": "whether to use the adafactor optimizer"}
    )
    top_k: Optional[float] = field(default=0.0, metadata={"help": "Value for top_k"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Value for top_p"})
    do_sample: Optional[bool] = field(
        default=True, metadata={"help": "Flag for sampling"}
    )
    eos_token_id: Optional[int] = field(
        default=100_000, metadata={"help": "End of sentence token id"}
    )
    output_min_length: Optional[int] = field(
        default=32, metadata={"help": "maximum length for generation"}
    )
    output_max_length: Optional[int] = field(
        default=128, metadata={"help": "maximum length for generation"}
    )
    mini_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the PPO minibatch size"}
    )
    ppo_batch_size: Optional[int] = field(
        default=8, metadata={"help": "the PPO batch size"}
    )
    ppo_epochs: Optional[int] = field(
        default=10, metadata={"help": "the number of optimisation epochs per batch of samples"}
    )
    total_epochs: Optional[int] = field(
        default=100, metadata={"help": "number of total epochs"}
    )
    early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "whether to early stop"}
    )
    target_kl: Optional[float] = field(
        default=0.1, metadata={"help": "kl target for early stopping"}
    )
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "Initial KL penalty coefficient (used for adaptive and linear control)"
        },
    )
    adap_kl_ctrl: Optional[bool] = field(
        default=True, metadata={"help": "Use adaptive KL control, otherwise linear"}
    )
    rl_lora_path: Optional[str] = field(
        default="step1_reinforcement_learning_final_lora_weights/",
        metadata={
            "help": "Output directory for step 1 supervised finetuning's Lora weights."
        },
    )
