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


import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import evaluate
import numpy as np
import torch
from pykoi.db.qa_database import QuestionAnswerDatabase
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model

from tqdm import tqdm
from transformers import (Adafactor, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainerCallback, TrainingArguments, logging,
                          pipeline, set_seed)
from transformers.utils import PushToHubMixin
from trl import (AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer,
                 SFTTrainer)
from trl.core import LengthSampler
from trl.trainer.utils import ConstantLengthDataset, PeftSavingCallback
from pykoi.rlhf.config import RLHFConfig

# from pykoi.db.ranking_database import (
#     QA_CSV_HEADER,
#     QA_CSV_HEADER_ID,
#     QA_CSV_HEADER_QUESTION,
#     QA_CSV_HEADER_ANSWER,
#     QA_CSV_HEADER_VOTE_STATUS,
# )
QA_CSV_HEADER_ID = 'ID'
QA_CSV_HEADER_QUESTION = 'Question'
QA_CSV_HEADER_ANSWER = 'Answer'
QA_CSV_HEADER_VOTE_STATUS = 'Vote Status'
QA_CSV_HEADER_TIMESTAMPS = 'Timestamp'
QA_CSV_HEADER = (
    QA_CSV_HEADER_ID,
    QA_CSV_HEADER_QUESTION,
    QA_CSV_HEADER_ANSWER,
    QA_CSV_HEADER_VOTE_STATUS,
    QA_CSV_HEADER_TIMESTAMPS
)

def read_json_file(file_path):
    """
    Reads a JSON file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file as a dictionary.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


############################################################
## Step 1: Supervised Finetuning Trainer
############################################################


class SFT(Trainer):
    """
    A class representing the supervised finetuning trainer.

    Attributes:
        rlhf_config (RLHFConfig): The RLHF configuration object.
        tokenizer (AutoTokenizer): The tokenizer used for tokenizing the input data.
        num_proc (int): The number of workers to use for data loading.
        dataset (Dict[str, Dataset]): A dictionary containing the train and eval datasets.
        torch_dtype (torch.dtype): The torch data type to use for training.
        training_args (TrainingArguments): The training arguments for the trainer.
        model (AutoModelForCausalLM): The model to train.
        trainer (SFTTrainer): The trainer object used for training the model.
    """

    def __init__(self, rlhf_config: RLHFConfig):
        """
        Initializes the SFTTrainer object.

        Args:
            rlhf_config (RLHFConfig): The RLHF configuration object.
        """
        self._rlhf_config = rlhf_config
        self.tokenizer = AutoTokenizer.from_pretrained(rlhf_config.base_model_path)
        self.num_proc = (
            self._rlhf_config.num_workers if not self._rlhf_config.streaming else None
        )
        self.dataset = self.create_datasets(self.tokenizer, self._rlhf_config)
        self.torch_dtype = torch.bfloat16 if self._rlhf_config.bf16 else torch.float16
        # self.torch_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
        self.training_args = TrainingArguments(
            output_dir=self._rlhf_config.output_dir,
            dataloader_drop_last=True,
            evaluation_strategy=self._rlhf_config.evaluation_strategy,
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
            trust_remote_code=True,
            load_in_8bit=self._rlhf_config.load_in_8bit,
            device_map=self._rlhf_config.device_map,
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
        """
        Trains the model using the SFTTrainer object.
        """
        self.trainer.train()

    def load_lora(self, base_model_path=None, lora_model_path=None):

        if base_model_path is None:
            base_model_path = self._rlhf_config.base_model_path

        ## Load lora config
        if lora_model_path is None:
            lora_config = self.trainer.model.config
        else:
            lora_config = PeftConfig.from_pretrained(lora_model_path)

        ## Load the base tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if lora_config.task_type == "SEQ_CLS":
            # peft is for reward model so load sequence classification
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_path, num_labels=1, torch_dtype=self._rlhf_config.torch_dtype
            )
        elif lora_config.task_type == "CAUSAL_LM":
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                return_dict=True,
                torch_dtype=self._rlhf_config.torch_dtype,
            )
        else:
            raise ValueError("Invalid task_type in lora_config")

        # Merge the base model and the Lora model
        model = PeftModel.from_pretrained(base_model, lora_config)
        return model, tokenizer

    def save(self, output_path=None):

        if output_path is None:
            output_path = os.path.join(
                self._rlhf_config.output_dir, self._rlhf_config.sft_lora_path
            )
        self.trainer.save_model(output_path)

    def train_and_save(self, output_path=None):
        self.trainer.train()
        self.save(output_path)

    def prepare_sample_text(self, example):
        """Prepare the text from a sample of the dataset."""
        text = f"Question: {example[self._rlhf_config.question_title]}\n\n\
            Answer: {example[self._rlhf_config.answer_title]}"
        return text

    def create_datasets(self, tokenizer, args):
        if args.dataset_type == "local_db":
            qa_database = QuestionAnswerDatabase()
            my_data_pd = qa_database.retrieve_all_question_answers_as_pandas()
            my_data_pd = my_data_pd[my_data_pd[QA_CSV_HEADER_VOTE_STATUS]=="up"]
            my_data_pd = my_data_pd[[QA_CSV_HEADER_ID,
                                     QA_CSV_HEADER_QUESTION,
                                     QA_CSV_HEADER_ANSWER]]
            print("My local database has {} samples".format(my_data_pd.shape[0]))
            dataset = Dataset.from_dict(my_data_pd)
        elif args.dataset_type == "local_csv":
            dataset = load_dataset('csv', data_files=args.dataset_name)
            dataset = dataset[args.split] # Convert DatasetDict to Dataset
        elif args.dataset_type == "huggingface":
            dataset = load_dataset(
                args.dataset_name,
                data_dir=args.dataset_subset_sft,
                split=args.split,
                use_auth_token=True,
                num_proc=self.num_proc,
                streaming=args.streaming,
            )
            dataset = dataset[args.split] # Convert DatasetDict to Dataset
        else:
            raise FileNotFoundError(f"No (supported) data files or dataset script found {args.dataset_type}")
        
        dataset = dataset.train_test_split(test_size=args.train_test_split_ratio, 
                                           seed=args.seed)
        print(f"Size of the train set: {len(dataset['train'])}. \
              Size of the validation set: {len(dataset['test'])}")

        train_dataset = ConstantLengthDataset(
            tokenizer,
            dataset["train"],
            formatting_func=self.prepare_sample_text,
            infinite=True,
            seq_length=args.max_seq_length,
            # chars_per_token=chars_per_token,
        )
        eval_dataset = ConstantLengthDataset(
            tokenizer,
            dataset["test"],
            formatting_func=self.prepare_sample_text,
            infinite=False,
            seq_length=args.max_seq_length,
            # chars_per_token=chars_per_token,
        )
        return {"train": train_dataset, "eval": eval_dataset}


############################################################
## Step 2: Reward Trainer
############################################################

## TODO: LOAD FROM THE DATABASE.PY FILE
RANKING_CSV_HEADER_ID = "ID"
RANKING_CSV_HEADER_QUESTION = "Question"
RANKING_CSV_HEADER_UP_RANKING_ANSWER = "Up Ranking Answer"
RANKING_CSV_HEADER_LOW_RANKING_ANSWER = "Low Ranking Answer"
RANKING_CSV_HEADER = (
    RANKING_CSV_HEADER_ID,
    RANKING_CSV_HEADER_QUESTION,
    RANKING_CSV_HEADER_UP_RANKING_ANSWER,
    RANKING_CSV_HEADER_LOW_RANKING_ANSWER,
)

# We need to define a special data collator that batches the ranking data.
@dataclass
class RewardDataCollatorWithPadding:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length: Optional[int] = max_length
        self.padding = True
        self.pad_to_multiple_of: Optional[int] = None
        self.return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        def extract_and_pad(key_ids, key_mask):
            extracted_features = [
                {"input_ids": f[key_ids], "attention_mask": f[key_mask]}
                for f in features
            ]
            return self.tokenizer.pad(
                extracted_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

        batch_x = extract_and_pad("input_ids_x", "attention_mask_x")
        batch_y = extract_and_pad("input_ids_y", "attention_mask_y")

        return {
            "input_ids_x": batch_x["input_ids"],
            "attention_mask_x": batch_x["attention_mask"],
            "input_ids_y": batch_y["input_ids"],
            "attention_mask_y": batch_y["attention_mask"],
            "return_loss": True,
        }


class RewardTrainer(Trainer):
    def __init__(self, rlhf_config: RLHFConfig):
        self._rlhf_config = rlhf_config
        self.args = TrainingArguments(
            output_dir=rlhf_config.output_dir,
            learning_rate=rlhf_config.learning_rate,
            per_device_train_batch_size=rlhf_config.per_device_train_batch_size,
            per_device_eval_batch_size=rlhf_config.per_device_eval_batch_size,
            num_train_epochs=rlhf_config.reward_epochs,
            weight_decay=rlhf_config.weight_decay,
            evaluation_strategy=rlhf_config.evaluation_strategy,
            save_strategy=rlhf_config.evaluation_strategy,
            gradient_accumulation_steps=rlhf_config.gradient_accumulation_steps,
            gradient_checkpointing=rlhf_config.gradient_checkpointing,
            # deepspeed=rlhf_config.deepspeed,
            local_rank=rlhf_config.local_rank,
            remove_unused_columns=rlhf_config.remove_unused_columns,
            label_names=rlhf_config.label_names,
            bf16=rlhf_config.bf16,
            logging_strategy=rlhf_config.logging_strategy,
            logging_steps=rlhf_config.logging_steps,
            # optim=rlhf_config.optim,
            # lr_scheduler_type=rlhf_config.lr_scheduler_type_rw
        )
        self.torch_dtype = torch.bfloat16 if rlhf_config.bf16 else torch.float16
        # self.torch_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)

        ## Load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(rlhf_config.reward_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            rlhf_config.reward_model_path,
            num_labels=1,
            torch_dtype=self.torch_dtype,
            load_in_8bit=rlhf_config.load_in_8bit,
            device_map=rlhf_config.device_map,
        )
        self.model = get_peft_model(self.base_model, rlhf_config.lora_config_reward)
        self.model.print_trainable_parameters()
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.use_cache = not rlhf_config.gradient_checkpointing
        self.num_proc = (
            self._rlhf_config.num_workers if not self._rlhf_config.streaming else None
        )

        self.dataset = self.create_datasets()
        self.dataset = self.create_datasets()

        self.compute_metrics = self._compute_metrics
        self.data_collator = RewardDataCollatorWithPadding(
            tokenizer=self.tokenizer, max_length=self._rlhf_config.max_seq_length_reward
        )
        super().__init__(
            model=self.model,
            args=self.args,
            data_collator=self.data_collator,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["eval"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

    def _preprocess_function(self, examples):
        """
        Turn the dataset into pairs of question and answer, where
        "text_x = question + preferred answer" and "text_y = question + not preferred answer".
        Then tokenize the dataset.

        Returns:
            A dictionary containing the processed data.
        """

        def tokenize_and_store(question, answer, key_ids, key_mask):
            tokenized = self.tokenizer(
                f"Question: {question}\n\nAnswer: {answer}", truncation=True
            )
            new_examples[key_ids].append(tokenized["input_ids"])
            new_examples[key_mask].append(tokenized["attention_mask"])

        new_examples = {
            "input_ids_x": [],
            "attention_mask_x": [],
            "input_ids_y": [],
            "attention_mask_y": [],
        }

        for question, better_answer, worse_answer in zip(
            examples[RANKING_CSV_HEADER_QUESTION],
            examples[RANKING_CSV_HEADER_UP_RANKING_ANSWER],
            examples[RANKING_CSV_HEADER_LOW_RANKING_ANSWER],
        ):
            tokenize_and_store(
                question, better_answer, "input_ids_x", "attention_mask_x"
            )
            tokenize_and_store(
                question, worse_answer, "input_ids_y", "attention_mask_y"
            )

        return new_examples

    def create_datasets(self):
        """
        Load the dataset and preprocess it.

        Returns:
            A dictionary containing the train and eval datasets.
        """
        ## based on dataset_type (e.g. "huggingface", "csv", etc.), load the data
        if self._rlhf_config.dataset_type == "huggingface":
            dataset = load_dataset(
                self._rlhf_config.dataset_name,
                data_dir=self._rlhf_config.dataset_reward_folder,
                split=self._rlhf_config.split,
                use_auth_token=True,
                num_proc=self._rlhf_config.num_proc,
                streaming=self._rlhf_config.streaming,
            )
        elif self._rlhf_config.dataset_type == "csv":
            dataset = load_dataset("csv", data_files=self._rlhf_config.dataset_name)
        else:
            raise FileNotFoundError(
                f"No (supported) data files or dataset script found {self._rlhf_config.dataset_type}"
            )

        # Preprocess the dataset and filter out QAs that are longer than max_length
        dataset = dataset.map(
            self._preprocess_function,
            batched=True,
            num_proc=self.num_proc,
        )
        dataset = dataset.filter(
            lambda x: len(x["input_ids_x"]) <= self._rlhf_config.max_seq_length_reward
            and len(x["input_ids_y"]) <= self._rlhf_config.max_seq_length_reward
        )

        dataset = dataset[self._rlhf_config.split]  # Convert DatasetDict to Dataset
        ## load desired amount of data
        if self._rlhf_config.reward_num_of_data > 0:
            dataset = dataset.select(
                range(min(self._rlhf_config.reward_num_of_data, len(dataset)))
            )

        ## split to train and test datasets
        dataset = dataset.train_test_split(
            test_size=self._rlhf_config.train_test_split_ratio,
            seed=self._rlhf_config.seed,
        )
        print(
            f"Size of the train set: {len(dataset['train'])}. \
                Size of the validation set: {len(dataset['test'])}"
        )

        return {"train": dataset["train"], "eval": dataset["test"]}

    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss:
    # https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_x = model(
            input_ids=inputs["input_ids_x"], attention_mask=inputs["attention_mask_x"]
        )[0]
        rewards_y = model(
            input_ids=inputs["input_ids_y"], attention_mask=inputs["attention_mask_y"]
        )[0]
        loss = -torch.nn.functional.logsigmoid(rewards_x - rewards_y).mean()
        if return_outputs:
            return loss, {"rewards_x": rewards_x, "rewards_y": rewards_y}
        return loss

    def _compute_metrics(self, eval_pred):
        """
        Compute the accuracy of the model.

        Args:
            eval_pred (:obj:`EvalPrediction`): The evaluation prediction.

        Returns:
            A dictionary containing the accuracy.
        """
        predictions, _ = eval_pred
        # Here, predictions is rewards_x and rewards_y.
        # We want to see how much of the time rewards_x > rewards_y.
        predictions = np.argmax(predictions, axis=0)
        labels = np.zeros(predictions.shape)
        accuracy = evaluate.load("accuracy")
        return accuracy.compute(predictions=predictions, references=labels)

    def save(self, output_path=None):
        """
        Save the model.

        Args:
            output_path (:obj:`str`, `optional`): The output path to save the model. If not provided, the model will be
                saved to the default output path.
        """
        if output_path is None:
            output_path = os.path.join(
                self._rlhf_config.output_dir, self._rlhf_config.reward_merged_path
            )
        self.save_model(output_path)

    def train_and_save(self, output_path=None):
        """
        Train the model and save it.

        Args:
            output_path (:obj:`str`, `optional`): The output path to save the model. If not provided, the model will be
                saved to the default output path.
        """
        self.train()
        self.save(output_path)


############################################################
## Step 3: RL Trainer
############################################################


class RL(Trainer):
    def __init__(self, rlhf_config: RLHFConfig):
        self._rlhf_config = rlhf_config
        self.accelerator = Accelerator()
        self.num_proc = self._rlhf_config.num_workers if not self._rlhf_config.streaming else None
        set_seed(rlhf_config.seed) ## TODO: how to set seed properly in __init__?

        self.ppo_config=PPOConfig(
            steps=self._rlhf_config.total_ppo_epochs,
            model_name=self._rlhf_config.base_model_path,
            learning_rate=self._rlhf_config.learning_rate,
            batch_size=self._rlhf_config.ppo_batch_size,
            mini_batch_size=self._rlhf_config.mini_batch_size,
            gradient_accumulation_steps=self._rlhf_config.gradient_accumulation_steps,
            optimize_cuda_cache=True,
            early_stopping=self._rlhf_config.early_stopping,
            target_kl=self._rlhf_config.target_kl,
            ppo_epochs=self._rlhf_config.ppo_epochs,
            seed=self._rlhf_config.seed,
            init_kl_coef=self._rlhf_config.init_kl_coef,
            adap_kl_ctrl=self._rlhf_config.adap_kl_ctrl,
            accelerator_kwargs=self._rlhf_config.accelerator_kwargs,
            )
        
        ## Load the base model and tokenizer and define the PPO Trainer for RL
        self.base_tokenizer = self.create_tokenizer(rlhf_config.base_model_path)
        self.base_dataset=self.create_dataset(self.base_tokenizer)
        self.base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            rlhf_config.base_model_path,
            load_in_8bit=rlhf_config.load_in_8bit,
            # is_loaded_in_8bit = True, # TODO TypeError: LlamaForCausalLM.__init__() got an unexpected keyword argument 'is_loaded_in_8bit'
            # torch_dtype=torch.float16, 
            device_map={"": Accelerator().local_process_index},
            peft_config=rlhf_config.lora_config_rl, 
        )
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.base_model,
            ref_model=None,
            tokenizer=self.base_tokenizer,
            dataset=self.base_dataset,
            data_collator=self.data_collator,
            # optimizer=optimizer,
            # peft_config=lora_config, ## PPOTrainer doesn't support parameter peft_config
        )
        self.base_kwargs = {
            # "min_length": -1,
            "top_k": rlhf_config.top_k,
            "top_p": rlhf_config.top_p,
            "do_sample": rlhf_config.do_sample,
            "pad_token_id": self.base_tokenizer.pad_token_id,
            "eos_token_id": rlhf_config.eos_token_id,
            "max_length": rlhf_config.output_max_length
        }

        ## Load the reward model and tokenizer and define the reward pipeline
        self.reward_tokenizer = self.create_tokenizer(rlhf_config.reward_model_path)
        self.reward_dataset=self.create_dataset(self.reward_tokenizer)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            rlhf_config.reward_model_path, 
            num_labels=1,
            # torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            device_map={"": Accelerator().local_process_index}
        )
        self.reward_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": self._rlhf_config.ppo_batch_size,
            "truncation": True,
            "max_length": self._rlhf_config.output_max_length
        }
        self.reward_pipe = pipeline(
            "sentiment-analysis",
            model=self.reward_model,
            # device_map={"": Accelerator().local_process_index},
            # model_kwargs={"load_in_8bit": True},
            model_kwargs=self.reward_kwargs,
            tokenizer=self.reward_tokenizer,
            return_token_type_ids=False,
        )

        
    def create_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


    def data_collator(self, data):
        return dict((key, [d[key] for d in data]) for key in data[0])


    def create_dataset(self, tokenizer):
        """
        Build dataset for training. This builds the dataset from `load_dataset`, one should
        customize this function to train the model on its own dataset.
        """
        ds = load_dataset(self._rlhf_config.dataset_name, 
                          data_dir=self._rlhf_config.dataset_subset_rl, 
                          split=self._rlhf_config.split)
        ds = ds.select(range(self._rlhf_config.dataset_subset_rl_train))

        def preprocess_function(examples):
            queries = ["Question: " + q + "\n\nAnswer: " for q in examples["question"]]
            input_ids = [tokenizer(q, truncation=True)["input_ids"] for q in queries]
            return {"query": queries, "input_ids": input_ids}

        ds = ds.map(
            preprocess_function,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=ds.column_names,
        )
        ds = ds.filter(lambda x: len(x["input_ids"]) < self._rlhf_config.max_seq_length, 
                       batched=False)
        ds.set_format(type="torch") ## TODO

        return ds


    def train(self):
        for epoch, batch in tqdm(enumerate(self.ppo_trainer.dataloader)):
            if epoch >= self._rlhf_config.total_ppo_epochs:
                break
            ## embed the questions and responses to tensors
            question_tensors = batch["input_ids"]
            response_tensors = self.ppo_trainer.generate(
                question_tensors,
                return_prompt=False,
                length_sampler=LengthSampler(self._rlhf_config.output_min_length, 
                                             self._rlhf_config.output_max_length),
                **self.base_kwargs,
            )
            batch["response"] = self.base_tokenizer.batch_decode(response_tensors, 
                                                                 skip_special_tokens=True)
            # compute rewards and run PPO
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = self.reward_pipe(texts, **self.reward_kwargs)
            rewards = [torch.tensor(output[0]["score"] - self._rlhf_config.reward_baseline) \
                       for output in pipe_outputs]
            stats = self.ppo_trainer.step(question_tensors, response_tensors, rewards)
            self.ppo_trainer.log_stats(stats, batch, rewards)

            ## save weights
            if self._rlhf_config.save_freq and epoch and \
                epoch % self._rlhf_config.save_freq == 0:
                self.ppo_trainer.save_pretrained(
                    os.path.join(self._rlhf_config.output_dir, f"rlhf_rl_step_{epoch}"))
