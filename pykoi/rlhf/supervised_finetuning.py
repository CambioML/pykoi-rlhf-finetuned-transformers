"""superised_finetuning."""
import os
import time
from datetime import datetime
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from peft import PeftConfig, PeftModel
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          TrainingArguments)
from trl import SFTTrainer
from trl.trainer.utils import ConstantLengthDataset

from pykoi.chat.db.constants import (QA_CSV_HEADER_ANSWER, QA_CSV_HEADER_ID,
                                     QA_CSV_HEADER_QUESTION,
                                     QA_CSV_HEADER_VOTE_STATUS)
from pykoi.chat.db.qa_database import QuestionAnswerDatabase
from pykoi.rlhf.config import RLHFConfig
from pykoi.telemetry.events import SFTStartEvent, SFTStopEvent
from pykoi.telemetry.telemetry import Telemetry
from pykoi.rlhf.customize_data_collator import DataCollatorForCompletionOnlyLM

class SupervisedFinetuning:
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

    def __init__(self, rlhf_config: RLHFConfig, enable_telemetry: bool = True) -> None:
        """
        Initializes the SFTTrainer object.

        Args:
            rlhf_config (RLHFConfig): The RLHF configuration object.
            enbale_telemetry (bool): Whether to enable telemetry or not.
        """
        self._telemetry = Telemetry(enable_telemetry)
        self._rlhf_config = rlhf_config
        self.tokenizer = AutoTokenizer.from_pretrained(rlhf_config.base_model_path)
        # dh: add special tokens to tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        END_KEY = "### End"
        INSTRUCTION_KEY = "### Instruction:"
        RESPONSE_KEY = "### Response:"
        RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
        self.tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})        
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
            gradient_checkpointing_kwargs={
                "use_reentrant": self._rlhf_config.gradient_checkpointing_use_reentrant
            },
            fp16=self._rlhf_config.fp16,
            bf16=self._rlhf_config.bf16,
            weight_decay=self._rlhf_config.weight_decay,
            run_name="step1_supervised_finetuning",
            ddp_find_unused_parameters=False,
            num_train_epochs=self._rlhf_config.num_train_epochs,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self._rlhf_config.base_model_path,
            load_in_8bit=self._rlhf_config.load_in_8bit,
            device_map=self._rlhf_config.device_map,
        )
        # resize the token embeddings to include the added special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        # dh: try the customized data collator that only predicts the answer part
        data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=self.tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["eval"],
            peft_config=self._rlhf_config.lora_config_rl, ## TODO: DH: LoraConfig MAY BE IGNORED IF USING FROM_PRETRAINED
            packing=True,
            data_collator=data_collator,
            dataset_text_field="text",
        )

    def train(self):
        """
        Trains the model using the SFTTrainer object.
        """
        self.trainer.train()

    def load_lora(
        self,
        base_model_path: Optional[str] = None,
        lora_model_path: Optional[str] = None,
    ):
        #import pdb; pdb.set_trace()
        # dh: not used
        if base_model_path is None:
            base_model_path = self._rlhf_config.base_model_path

        # Load lora config
        if lora_model_path is None:
            lora_config = self.trainer.model.config
        else:
            lora_config = PeftConfig.from_pretrained(lora_model_path)

        # Load the base tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if lora_config.task_type == "SEQ_CLS":
            # peft is for reward model so load sequence classification
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_path,
                num_labels=1,
                torch_dtype=self._rlhf_config.torch_dtype,
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
        start_event = SFTStartEvent(
            start_time=time.time(), date_time=datetime.utcfromtimestamp(time.time())
        )
        self._telemetry.capture(start_event)
        self.trainer.train()
        self.save(output_path)
        self._telemetry.capture(
            SFTStopEvent(
                end_time=time.time(),
                date_time=datetime.utcfromtimestamp(time.time()),
                duration=time.time() - start_event.start_time,
            )
        )

    def prepare_sample_text(self, example):
        """Prepare the text from a sample of the dataset."""
        text = (
            f"Question: {example[self._rlhf_config.question_title]}\n\n        "
            f"    Answer: {example[self._rlhf_config.answer_title]}"
        )
        return text
    

    def prepare_d2l_text(self, example):
        """Prepare the text from a sample of the d2l dataset ."""
        INTRO_BLURB = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        )
        INSTRUCTION_KEY = "### Instruction:"
        INPUT_KEY = "Input:"
        RESPONSE_KEY = "### Response:"
        END_KEY = "### End"
        RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
        DEFAULT_SEED = 42

        # This is a training prompt that does not contain an input string.  The instruction by itself has enough information
        # to respond.  For example, the instruction might ask for the year a historic figure was born.
        PROMPT_NO_INPUT_FORMAT = """{intro}
        {instruction_key}
        {instruction}
        {response_key}
        {response}
        {end_key}""".format(
            intro=INTRO_BLURB,
            instruction_key=INSTRUCTION_KEY,
            instruction="{instruction}",
            response_key=RESPONSE_KEY,
            response="{response}",
            end_key=END_KEY,
        )

        # This is a training prompt that contains an input string that serves as context for the instruction.  For example,
        # the input might be a passage from Wikipedia and the intruction is to extract some information from it.
        PROMPT_WITH_INPUT_FORMAT = """{intro}
        {instruction_key}
        {instruction}
        {input_key}
        {input}
        {response_key}
        {response}
        {end_key}""".format(
            intro=INTRO_BLURB,
            instruction_key=INSTRUCTION_KEY,
            instruction="{instruction}",
            input_key=INPUT_KEY,
            input="{input}",
            response_key=RESPONSE_KEY,
            response="{response}",
            end_key=END_KEY,
        )
        
        context = example.get("context")
        if context:
            text = PROMPT_WITH_INPUT_FORMAT.format(instruction=example["instruction"], response=example["response"], input=context)
        else:
            text = PROMPT_NO_INPUT_FORMAT.format(instruction=example["instruction"], response=example["instruction"])



        return text

    def create_datasets(self, tokenizer, args):
        if args.dataset_type == "local_db":
            qa_database = QuestionAnswerDatabase()
            my_data_pd = qa_database.retrieve_all_question_answers_as_pandas()
            my_data_pd = my_data_pd[my_data_pd[QA_CSV_HEADER_VOTE_STATUS] == "up"]
            my_data_pd = my_data_pd[
                [QA_CSV_HEADER_ID, QA_CSV_HEADER_QUESTION, QA_CSV_HEADER_ANSWER]
            ]
            print(
                "My local database has {} up vote samples for SFT".format(
                    my_data_pd.shape[0]
                )
            )
            dataset = Dataset.from_dict(my_data_pd)
        elif args.dataset_type == "local_csv":
            ## this way will load 1660 enetries
            # dataset = load_dataset("csv", data_files=args.dataset_name)
            # dataset = dataset[args.split]  # Convert DatasetDict to Dataset

            # this way will load 166 entries

            dataset = load_dataset("csv", data_files=args.dataset_name, split='train[:10%]')

        elif args.dataset_type == "huggingface":
            dataset = load_dataset(
                args.dataset_name,
                data_dir=args.dataset_subset_sft,
                split=args.split,
                use_auth_token=True,
                num_proc=self.num_proc,
                streaming=args.streaming,
            )
            dataset = dataset[args.split]  # Convert DatasetDict to Dataset
        else:
            raise FileNotFoundError(
                "No (supported) data files or dataset script found"
                f" {args.dataset_type}"
            )
        
        # dh: temp change. No test set
        # dataset = dataset.train_test_split(
        #     test_size=args.train_test_split_ratio, seed=args.seed
        # )
        print(
            f"Size of the train set: {len(dataset)}.              "
            #f"Size of the train set: {len(dataset['train'])}.              "
            #f" Size of the validation set: {len(dataset['test'])}"
        )

        train_dataset = ConstantLengthDataset(
            tokenizer,
            dataset,
            #dataset["train"], #dh: temp change. No test set
            formatting_func=self.prepare_d2l_text,
            infinite=True,
            seq_length=args.max_seq_length,
            # chars_per_token=chars_per_token,
        )
        # temp change: no test set
        # eval_dataset = ConstantLengthDataset(
        #     tokenizer,
        #     dataset["test"],
        #     formatting_func=self.prepare_d2l_text,
        #     infinite=False,
        #     seq_length=args.max_seq_length,
        #     # chars_per_token=chars_per_token,
        # )
        eval_dataset = None
        return {"train": train_dataset, "eval": eval_dataset}
