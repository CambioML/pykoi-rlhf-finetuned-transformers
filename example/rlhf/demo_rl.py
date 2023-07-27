<<<<<<< HEAD
## pip install xformers ## TODO
import pdb; 

=======
>>>>>>> 6f91a55284f1664bc0bfc9459d56e1a338a838c1
import pykoi
from pykoi import RLHFConfig

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import evaluate
import numpy as np
import torch
from pykoi.db.qa_database import QuestionAnswerDatabase
from accelerate import Accelerator, notebook_launcher
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
            # accelerator_kwargs=self._rlhf_config.accelerator_kwargs,
            )
        
<<<<<<< HEAD
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

        
        ## Load the base model and tokenizer and define the PPO Trainer for RL
        self.base_tokenizer = self.create_tokenizer(rlhf_config.base_model_path)
        pdb.set_trace()
=======
        ## Load the base model and tokenizer and define the PPO Trainer for RL
        self.base_tokenizer = self.create_tokenizer(rlhf_config.base_model_path)
>>>>>>> 6f91a55284f1664bc0bfc9459d56e1a338a838c1
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

<<<<<<< HEAD
=======
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
>>>>>>> 6f91a55284f1664bc0bfc9459d56e1a338a838c1

        
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
        args = self._rlhf_config
        if args.dataset_type == "local_db":
            qa_database = QuestionAnswerDatabase(db_file=self._rlhf_config.dataset_name)
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
                data_dir=args.dataset_subset_rl,
                split=args.split,
                use_auth_token=True,
                # num_proc=self.num_proc,
                # streaming=args.streaming,
            )
        ## TODO: if args.split in dataset.columns:
            # dataset = dataset[args.split] # Convert DatasetDict to Dataset
        else:
            raise FileNotFoundError(f"No (supported) data files or dataset script found {args.dataset_type}")
        
        # dataset = dataset.train_test_split(test_size=args.train_test_split_ratio, 
        #                                    seed=args.seed)
        # print(f"Size of the train set: {len(dataset['train'])}. \
        #       Size of the validation set: {len(dataset['test'])}")
        
        # dataset = dataset.select(range(self._rlhf_config.dataset_subset_rl_train))

        def preprocess_function(examples):
            new_examples = {
                "query": [],
                "input_ids": [],
            }
            for question in examples[QA_CSV_HEADER_QUESTION]:
                query = "Question: " + question + "\n\nAnswer: "
                tokenized_question = tokenizer(query, truncation=True)
                new_examples["query"].append(query)
                new_examples["input_ids"].append(tokenized_question["input_ids"])
            return new_examples
<<<<<<< HEAD
        
=======
>>>>>>> 6f91a55284f1664bc0bfc9459d56e1a338a838c1
        # def preprocess_function(examples):
        #     queries = ["Question: " + q + "\n\nAnswer: " for q in examples[QA_CSV_HEADER_QUESTION]]
        #     input_ids = [tokenizer(q, truncation=True, ## TODO: max_length
        #                            max_length=self._rlhf_config.output_max_length)["input_ids"] for q in queries]
        #     return {"query": queries, "input_ids": input_ids}
        
<<<<<<< HEAD
        pdb.set_trace()
=======
>>>>>>> 6f91a55284f1664bc0bfc9459d56e1a338a838c1
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=24, ## TODO self.num_proc,
            remove_columns=dataset.column_names,
        )
        dataset = dataset.filter(lambda x: len(x["input_ids"]) < self._rlhf_config.max_seq_length, 
                       batched=False)
        dataset.set_format(type="torch") ## TODO

        return dataset


    def train(self):
        ## Initialize accelerator
        self.ppo_trainer.dataloader = self.accelerator.prepare(self.ppo_trainer.dataloader)

        ## training
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
            batch[QA_CSV_HEADER_ANSWER] = self.base_tokenizer.batch_decode(response_tensors, 
                                                                 skip_special_tokens=True)
            # compute rewards and run PPO
            texts = [q + r for q, r in zip(batch["query"], batch[QA_CSV_HEADER_ANSWER])]
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


config = pykoi.RLHFConfig(base_model_path="elinas/llama-7b-hf-transformers-4.29", # "meta-llama/Llama-2-7b-hf", 
                          dataset_type="huggingface", ## "local_db",
                          dataset_name="goldmermaid/stack_exchange_rank_10k_dataset", ##"lvwerra/stack-exchange-paired", ## "/home/ubuntu/git/pykoi/example/notebook/qd.db",
                          dataset_subset_rl = "data",
                          reward_model_path="goldmermaid/rlhf_reward_model",
                          save_freq=100,
                          ppo_batch_size=32,
                          ppo_epochs=1,
                          output_max_length=128
                          )
rlhf_step3_rl = RL(config)
<<<<<<< HEAD
rlhf_step3_rl.train() ## TODO output_path="./models/rlhf_step3_rl"
=======
rlhf_step3_rl.train("./models/rlhf_step3_rl")
>>>>>>> 6f91a55284f1664bc0bfc9459d56e1a338a838c1
