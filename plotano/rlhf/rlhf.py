# Copyright 2023 The CambioML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch

from accelerate import Accelerator
from datasets import load_dataset
from peft import (
    LoraConfig, 
    PeftConfig,
    PeftModel,
    TaskType,
    get_peft_model
)
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    AutoTokenizer, 
    Trainer,
    TrainerCallback,
    TrainingArguments, 
    logging, 
    pipeline,
    set_seed
)
from transformers.utils import PushToHubMixin
from trl import (
    AutoModelForCausalLMWithValueHead, 
    PPOConfig, 
    PPOTrainer, 
    SFTTrainer)
    
from trl.core import LengthSampler
from trl.trainer.utils import ConstantLengthDataset, PeftSavingCallback




def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


@dataclass
class RLHFConfig:
    base_model_path: str = field(
        default="elinas/llama-7b-hf-transformers-4.29", 
        metadata={"help": "Huggingface model name or a local path to the base model."})
    dataset_type: Optional[str] = field(
        default="csv",
        metadata={"help": "choose from 'csv', 'huggingface' to load the dataset."})
    dataset_name: str = field(
        default="lvwerra/stack-exchange-paired", 
        metadata={"help": "Huggingface dataset name or a local path to the dataset."})
    streaming: bool = field(
        default=False, 
        metadata={"help": "Whether to use streaming."})
    shuffle_buffer: int = field(
        default=5000, 
        metadata={"help": "Size of the shuffle buffer."})
    max_seq_length: int = field(
        default=512, 
        metadata={"help": "Maximum sequence length."})
    max_steps: int = field(
        default=1000, 
        metadata={"help": "Maximum number of training steps."})
    # batch_size: int = field(
    #     default=8, 
    #     metadata={"help": "Batch size."})
    per_device_train_batch_size: Optional[int] = field(
        default=2, metadata={"help": "Batch size per device for training."})
    per_device_eval_batch_size: Optional[int] = field(
        default=8, metadata={"help": "Batch size per device for evaluation."})
    gradient_accumulation_steps: int = field(
        default=4, 
        metadata={"help": "Number of steps for gradient accumulation."})
    eos_token_id: int = field(
        default=49152, 
        metadata={"help": "End-of-sequence token ID."})
    learning_rate: Optional[float] = field(
        default=1e-5, metadata={"help": "Learning rate."})
    weight_decay: Optional[float] = field(
        default=0.01, 
        metadata={"help": "Weight decay."})
    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu."})
    fp16: Optional[bool] = field(default=True, metadata={"help": "Enable FP16."})
    bf16: Optional[bool] = field(default=False, metadata={"help": "Enable BF16."})
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "Whether load the model weights in 8-bit or not."})
    # device_map ## TODO
    gradient_checkpointing: Optional[bool] = field(
        default=False, 
        metadata={"help": "Enable gradient checkpointing."})
    seed: Optional[int] = field(default=0, metadata={"help": "Random seed."})
    num_workers: Optional[int] = field(default=None, metadata={"help": "Number of workers."})
    output_dir: Optional[str] = field(
        default="./rlhf_checkpoints", 
        metadata={"help": "Output directory for all model weights."})
    log_freq: Optional[int] = field(default=1, metadata={"help": "Logging frequency."})
    eval_freq: Optional[int] = field(default=1000, metadata={"help": "Evaluation frequency."})
    save_freq: Optional[int] = field(default=1000, metadata={"help": "Model saving frequency."})
    push_to_hub: Optional[bool] = field(
        default=False, metadata={"help": "Whether push to Huggingface Hub or not."})
    
    ## Step 1 SFT parameters
    dataset_subset_sft: str = field(
        default="data/finetune", 
        metadata={"help": "Subset folder of the dataset to use."})
    dataset_subset_sft_train: Optional[int] = field(
        default=10000, 
        metadata={"help": "The size of the subset of the training data to use."})
    split: str = field(
        default="train", 
        metadata={"help": "Dataset split to use."})
    question_title: str = field(
        default="Question",
        metadata={"help": "the column name of questions from the database."}
    )
    answer_title: str = field(
        default="Answer",
        metadata={"help": "the column name of answers from the database."} ## TODO: e.g. response_j
    )
    size_valid_set: int = field(
        default=4000, 
        metadata={"help": "Size of the validation/eval set."})
    sft_lora_path: str = field(
        default="step1_supervised_finetuning_lora_final/", 
        metadata={"help": "Output directory for step 1 supervised finetuning's Lora weights."})
    sft_merged_path: str = field(
        default="step1_supervised_finetuning_merged/", 
        metadata={"help": "Output directory for step 1 supervised finetuning's merged weights."})
    lr_scheduler_type_sft: str = field(
        default="cosine", 
        metadata={"help": "Type of learning rate scheduler."})
    num_warmup_steps: int = field(
        default=100, 
        metadata={"help": "Number of warmup steps for the scheduler."})
    lora_config_rl: LoraConfig = field(
        default=LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
        metadata={"help": "LoRA configuration."},
    )

    ## Step 2 reward modeling parameters
    reward_model_path: Optional[str] = field(
        default="databricks/dolly-v2-3b", 
        metadata={"help": "Huggingface model name or a local path to the reward model."})
    reward_lora_path: str = field(
        default="step1_supervised_finetuning_lora_final/", 
        metadata={"help": "Output directory for step 1 supervised finetuning's Lora weights."})
    reward_merged_path: str = field(
        default="step1_supervised_finetuning_merged/", 
        metadata={"help": "Output directory for step 1 supervised finetuning's merged weights."})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    dataset_reward_train: Optional[str] = field(
        default="data/reward", 
        metadata={"help": "Subset folder of the reward dataset to use."})
    dataset_reward_eval: Optional[str] = field(
        default="data/evaluation", 
        metadata={"help": "Subset folder of the evaluation dataset to use."})
    dataset_subset_reward_train: Optional[int] = field(
        default=1000, 
        metadata={"help": "The size of the subset of the training data to use."})
    dataset_subset_reward_eval: Optional[int] = field(
        default=400, 
        metadata={"help": "The size of the subset of the validation/eval data to use."})
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "The number of training epochs."})
    # deepspeed: Optional[str] = field(
    #     default= "/home/ubuntu/peel-test/peel/peelml/rlhf/deepspeed_config.json", ## None, ## TODO
    #     metadata={"help": "Path to deepspeed config if using deepspeed."})
    remove_unused_columns: Optional[bool] = field(
        default=False, 
        metadata={"help": "Whether to remove unused columns from the dataset."})
    label_names: Optional[List[str]] = field(
        default_factory=list, 
        metadata={"help": "List of column names in the dataset to use as labels."})
    logging_strategy: Optional[str] = field(
        default="steps", 
        metadata={"help": "The strategy used for logging during training."})
    logging_steps: Optional[int] = field(
        default=10, 
        metadata={"help": "The number of steps between each logging."})
    # callbacks: Optional[List[TrainerCallback]] = field(
    #     default=[], ## PeftSavingCallback()
    #     metadata={"help": "The callbacks to use for training."}),
    # optim: Optional[str] = field(
    #     default="adamw_hf", metadata={"help": "The optimizer to use."})
    # lr_scheduler_type_rw: str = field(
    #     default="linear", 
    #     metadata={"help": "Type of learning rate scheduler."})
    lora_config_reward: LoraConfig = field(
        default=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
        ),
        metadata={"help": "LoRA configuration."},
    )

    ## Step 3 RL parameters
    dataset_subset_rl: str = field(
        default="data/finetune", 
        metadata={"help": "Subset folder of the dataset to use."})
    dataset_subset_rl_train: Optional[int] = field(
        default=10000, 
        metadata={"help": "The size of the subset of the training data to use."})
    adafactor: Optional[bool] = field(
        default=False, 
        metadata={"help": "whether to use the adafactor optimizer"})
    top_k: Optional[float] = field(
        default=0.0, metadata={"help": "Value for top_k"})
    top_p: Optional[float] = field(
        default=1.0, metadata={"help": "Value for top_p"})
    do_sample: Optional[bool] = field(
        default=True, metadata={"help": "Flag for sampling"})
    eos_token_id: Optional[int] = field(
        default=100_000, metadata={"help": "End of sentence token id"})
    output_max_length: Optional[int] = field(
        default=128, 
        metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the PPO minibatch size"})
    ppo_batch_size: Optional[int] = field(
        default=8, metadata={"help": "the PPO batch size"})
    ppo_epochs: Optional[int] = field(
        default=4, metadata={"help": "the number of ppo epochs"})
    total_ppo_epochs: Optional[int] = field(
        default=20000, metadata={"help": "number of epochs"}) 
    ## TODO: differences between total_ppo_epochs and ppo_epochs
    early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    adap_kl_ctrl: Optional[bool] = field(
        default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})


############################################################
## Step 1: Supervised Finetuning Trainer
############################################################

class SFT(Trainer):
    def __init__(self, rlhf_config: RLHFConfig):
        self._rlhf_config = rlhf_config
        self.tokenizer = AutoTokenizer.from_pretrained(rlhf_config.base_model_path)
        self.num_proc = self._rlhf_config.num_workers if not self._rlhf_config.streaming else None
        self.dataset = self.create_datasets(self.tokenizer, self._rlhf_config)

        self.training_args = TrainingArguments(
            output_dir=self._rlhf_config.output_dir,
            dataloader_drop_last=True,
            evaluation_strategy="steps",
            max_steps=self._rlhf_config.max_steps,
            eval_steps=self._rlhf_config.eval_freq,
            save_steps=self._rlhf_config.save_freq,
            logging_steps=self._rlhf_config.log_freq,
            per_device_train_batch_size=self._rlhf_config.per_device_train_batch_size,
            per_device_eval_batch_size=self._rlhf_config.per_device_eval_batch_size,
            learning_rate=self._rlhf_config.learning_rate,
            lr_scheduler_type=self._rlhf_config.lr_scheduler_type_sft,
            warmup_steps=self._rlhf_config.num_warmup_steps,
            gradient_accumulation_steps=self._rlhf_config.gradient_accumulation_steps,
            gradient_checkpointing=self._rlhf_config.gradient_checkpointing,
            fp16=self._rlhf_config.fp16,
            bf16=self._rlhf_config.bf16,
            weight_decay=self._rlhf_config.weight_decay,
            run_name="step1_supervised_finetuning",
            ddp_find_unused_parameters=False,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self._rlhf_config.base_model_path, 
            load_in_8bit=self._rlhf_config.load_in_8bit, 
            device_map={"": Accelerator().process_index} ## TODO
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["eval"],
            peft_config=self._rlhf_config.lora_config_rl,
            packing=True,
        )

    def train(self):
        self.trainer.train()

    def merge_lora(self, output_path=None):

        """
        # assuming your base model is the model with the pretrained weights 
        # and lora_model is the adapter model to be merged
        """

        ## Load the base model and tokenizer:
        base_model_name = self._rlhf_config.base_model_path
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # peft_config = PeftConfig.from_pretrained(lora_model)
        if self._rlhf_config.lora_config_rl.task_type == "SEQ_CLS":
            # peft is for reward model so load sequence classification
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name, num_labels=1, torch_dtype=torch.bfloat16 ## TODO
            )
        elif self._rlhf_config.lora_config_rl.task_type == "CAUSAL_LM": 
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, return_dict=True, torch_dtype=torch.bfloat16 ## TODO
            )
        else:
            raise ValueError("Invalid task_type in lora_config")


        # Merge the base model and the Lora model
        model = PeftModel.from_pretrained(base_model, self.trainer.model.config)
        # model.eval()
        model = model.merge_and_unload()

        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
                
        if self._rlhf_config.push_to_hub:
            model.push_to_hub(output_path, use_temp_dir=False)


    def save(self, output_path=None, merge_base_and_lora=False):    

        if output_path is None:
            output_path = os.path.join(
                self._rlhf_config.output_dir, 
                self._rlhf_config.sft_lora_path)
        # self.trainer.model.save_pretrained(output_path) ## only save "adapter_config.json" and "adapter_model.bin"
        self.trainer.save_model(output_path)

    def train_and_save(self, output_path=None, merge_weights=False):
        self.trainer.train()
        self.save(output_path, 
                  merge_base_and_lora=merge_weights)
        
        
    ## TODO: using source code from "supervised_finetuneing.py", need to rewrite to our own `utils.py`
    def prepare_sample_text(self, example):
        """Prepare the text from a sample of the dataset."""
        text = f"Question: {example[self._rlhf_config.question_title]}\n\n\
            Answer: {example[self._rlhf_config.answer_title]}"
        return text

    ## TODO: using source code from "supervised_finetuneing.py", need to rewrite to our own `utils.py`
    def create_datasets(self, tokenizer, args):
        if self._rlhf_config.dataset_type == "huggingface":
            dataset = load_dataset(
                args.dataset_name,
                data_dir=args.dataset_subset_sft,
                split=args.split,
                use_auth_token=True,
                num_proc=self.num_proc,
                streaming=args.streaming,
            )
        elif self._rlhf_config.dataset_type == "csv":
            dataset = load_dataset('csv', data_files=args.dataset_name) 
        else:
            raise FileNotFoundError(f"No (supported) data files or dataset script found {self._rlhf_config.dataset_type}")

        # if args.streaming:
        #     print("Loading the dataset in streaming mode")
        #     eval_data = dataset.take(args.size_valid_set)
        #     train_data = dataset.skip(args.size_valid_set)
        #     # if args.train_subset > 0:
        #     #     train_data = train_data.select(range(args.train_subset)) ## TODO
        #     train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
        # else:
        dataset = dataset[self._rlhf_config.split] # Convert DatasetDict to Dataset
        dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_data = dataset["train"]
        eval_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(eval_data)}")

        # chars_per_token = chars_token_ratio(train_data, tokenizer)
        # print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

        ## `ConstantLengthDataset` is used for efficient training: we concatenate a lot of 
        ## texts with a EOS token in between and cut chunks of the context size to fill 
        ## the batch without any padding.
        train_dataset = ConstantLengthDataset(
            tokenizer,
            train_data,
            formatting_func=self.prepare_sample_text,
            infinite=True,
            seq_length=args.max_seq_length,
            # chars_per_token=chars_per_token,
        )
        eval_dataset = ConstantLengthDataset(
            tokenizer,
            eval_data,
            formatting_func=self.prepare_sample_text,
            infinite=False,
            seq_length=args.max_seq_length,
            # chars_per_token=chars_per_token,
        )
        # return train_dataset, eval_dataset
        return {"train": train_dataset, "eval": eval_dataset}
