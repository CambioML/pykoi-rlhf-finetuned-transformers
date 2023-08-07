"""reward model finetuning."""
import os

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import evaluate
import numpy as np
import torch

from datasets import Dataset, load_dataset
from peft import get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from pykoi.rlhf.config import RLHFConfig
from pykoi.chat.db.constants import (
    RANKING_CSV_HEADER_ID,
    RANKING_CSV_HEADER_QUESTION,
    RANKING_CSV_HEADER_LOW_RANKING_ANSWER,
    RANKING_CSV_HEADER_UP_RANKING_ANSWER,
)
from pykoi.chat.db.ranking_database import RankingDatabase


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


class RewardFinetuning(Trainer):
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

        # Load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(
            rlhf_config.reward_model_path
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            rlhf_config.reward_model_path,
            num_labels=1,
            torch_dtype=self.torch_dtype,
            load_in_8bit=rlhf_config.load_in_8bit,
            device_map=rlhf_config.device_map,
        )
        self.model = get_peft_model(
            self.base_model, rlhf_config.lora_config_reward
        )
        self.model.print_trainable_parameters()
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.use_cache = not rlhf_config.gradient_checkpointing
        self.num_proc = (
            self._rlhf_config.num_workers
            if not self._rlhf_config.streaming
            else None
        )

        self.dataset = self.create_datasets()

        self.compute_metrics = self._compute_metrics
        self.data_collator = RewardDataCollatorWithPadding(
            tokenizer=self.tokenizer,
            max_length=self._rlhf_config.max_seq_length_reward,
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
        # based on dataset_type (e.g. "huggingface", "csv", etc.), load the data
        if self._rlhf_config.dataset_type == "local_db":
            ranking_database = RankingDatabase()
            my_data_pd = (
                ranking_database.retrieve_all_question_answers_as_pandas()
            )
            my_data_pd = my_data_pd[
                [
                    RANKING_CSV_HEADER_ID,
                    RANKING_CSV_HEADER_QUESTION,
                    RANKING_CSV_HEADER_UP_RANKING_ANSWER,
                    RANKING_CSV_HEADER_LOW_RANKING_ANSWER,
                ]
            ]
            print(
                "My local database has {} samples for RW finetuning".format(
                    my_data_pd.shape[0]
                )
            )
            dataset = Dataset.from_dict(my_data_pd)
        # elif self._rlhf_config.dataset_type == "huggingface":
        #     # TODO: get rid of this and this does not work
        #     dataset = load_dataset(
        #         self._rlhf_config.dataset_name,
        #         data_dir=self._rlhf_config.dataset_reward_folder,
        #         split=self._rlhf_config.split,
        #         use_auth_token=True,
        #         num_proc=self._rlhf_config.num_proc,
        #         streaming=self._rlhf_config.streaming,
        #     )
        elif self._rlhf_config.dataset_type == "csv":
            dataset = load_dataset(
                "csv", data_files=self._rlhf_config.dataset_name
            )
        else:
            raise FileNotFoundError(
                "No (supported) data files or dataset script found"
                f" {self._rlhf_config.dataset_type}"
            )

        # Preprocess the dataset and filter out QAs that are longer than max_length
        dataset = dataset.map(
            self._preprocess_function,
            batched=True,
            num_proc=self.num_proc,
        )
        dataset = dataset.filter(
            lambda x: len(x["input_ids_x"])
            <= self._rlhf_config.max_seq_length_reward
            and len(x["input_ids_y"]) <= self._rlhf_config.max_seq_length_reward
        )

        # dataset = dataset[self._rlhf_config.split]  #TODO # Convert DatasetDict to Dataset ## ='train'
        # load desired amount of data
        if self._rlhf_config.reward_num_of_data > 0:
            dataset = dataset.select(
                range(min(self._rlhf_config.reward_num_of_data, len(dataset)))
            )

        # split to train and test datasets
        dataset = dataset.train_test_split(
            test_size=self._rlhf_config.train_test_split_ratio,
            seed=self._rlhf_config.seed,
        )
        print(
            f"Size of the train set: {len(dataset['train'])}.                "
            f" Size of the validation set: {len(dataset['test'])}"
        )

        return {"train": dataset["train"], "eval": dataset["test"]}

    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss:
    # https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_x = model(
            input_ids=inputs["input_ids_x"],
            attention_mask=inputs["attention_mask_x"],
        )[0]
        rewards_y = model(
            input_ids=inputs["input_ids_y"],
            attention_mask=inputs["attention_mask_y"],
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
                self._rlhf_config.output_dir,
                self._rlhf_config.reward_merged_path,
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
