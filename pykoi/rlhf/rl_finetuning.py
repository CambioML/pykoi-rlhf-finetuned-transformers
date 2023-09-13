"""rl finetuning."""
import time
from datetime import datetime
from pykoi.rlhf.config import RLHFConfig
from pykoi.chat.db.constants import (
    QA_CSV_HEADER_ID,
    QA_CSV_HEADER_QUESTION,
    QA_CSV_HEADER_ANSWER,
    QA_CSV_HEADER_VOTE_STATUS,
)

import os
import json
import numpy as np
import torch
from pykoi.chat.db.qa_database import QuestionAnswerDatabase
from accelerate import Accelerator
from datasets import Dataset, load_dataset

from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    pipeline,
    set_seed,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig, AutoPeftModelForCausalLM
from pykoi.telemetry.telemetry import Telemetry
from pykoi.telemetry.events import (
    RLStartEvent,
    RLStopEvent,
)


class RLFinetuning(Trainer):
    def __init__(self,
                 rlhf_config: RLHFConfig,
                 enable_telemetry: bool = True) -> None:
        """
        RLFinetuning class for finetuning a language model using reinforcement learning.

        Args:
            rlhf_config (RLHFConfig): Configuration object for RLHF.
        """
        self._telemetry = Telemetry(enable_telemetry)
        self._rlhf_config = rlhf_config
        self.accelerator = Accelerator()
        self.num_proc = (
            self._rlhf_config.num_workers
            if not self._rlhf_config.streaming
            else None
        )
        set_seed(rlhf_config.seed)

        self.ppo_config = PPOConfig(
            steps=self._rlhf_config.total_epochs,
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

        ## Load the reward model and tokenizer and define the reward pipeline
        self.reward_tokenizer = self.create_tokenizer(
            rlhf_config.reward_model_path
        )
        self.reward_dataset = self.create_dataset(self.reward_tokenizer)

        reward_model_path = rlhf_config.reward_model_path

        try:
            # If there is a trained peft adapter in the hub, load its config.
            remote_adapter_config_reward = hf_hub_download(reward_model_path, "adapter_config.json")
        except:
            remote_adapter_config_reward = None


        local_adapter_present_reward =  os.path.exists(
            os.path.join(reward_model_path, "adapter_config.json")
        )

        # # Load the trained peft adapter config
        if local_adapter_present_reward:
            trained_adapter_config_reward = PeftConfig.from_pretrained(reward_model_path)
        else:
            trained_adapter_config = PeftConfig.from_pretrained(remote_adapter_config_reward)

        ## Load the pretrained base model
        pretrained_kwargs_reward = {
            "num_labels": 1,
            "load_in_8bit": False, #True,
            "device_map": {"": Accelerator().local_process_index},
            }   # TODO: ADD
        pretrained_model_reward = AutoModelForSequenceClassification.from_pretrained(
            trained_adapter_config_reward.base_model_name_or_path,
            **pretrained_kwargs_reward
        )
        ## TODO: LOAD MERGED BASE MODEL FROM STEP 2

        # Load the Peft model by combing the base model with the trained adapter
        reward_model = PeftModel.from_pretrained(pretrained_model_reward, reward_model_path, is_trainable=False) # TODO: fix this. This should not be trainable.
        self.reward_model = reward_model.merge_and_unload()
        #pretrained_model.print_trainable_parameters()
        print("\nTrained peft adapter loaded for reward model\n")
        # have to specify the pad_token_id or will lead to error: "Cannot handle batch sizes > 1 if no padding token is defined"
        # see https://stackoverflow.com/questions/68084302/assertionerror-cannot-handle-batch-sizes-1-if-no-padding-token-is-defined
        self.reward_model.config.pad_token_id = self.reward_tokenizer.pad_token_id
        



        self.reward_kwargs = {
            "top_k": None,
            "function_to_apply": "none",
            "batch_size": self._rlhf_config.ppo_batch_size,
            "truncation": True,
            "max_length": self._rlhf_config.output_max_length,
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
        self.base_dataset = self.create_dataset(self.base_tokenizer)

        pretrained_model_name_or_path = rlhf_config.base_model_path
        # #NOTE: TODO: peft config will be directly inferred from the pre-trained model. rlhf_config.lora_config_rl will be ignored in previous implementation. Do we want to use it, in the flow of using merged model as base model and then add peft adapter again?? 

        pretrained_kwargs = {
            "load_in_8bit": rlhf_config.load_in_8bit,
            "device_map": {"": Accelerator().local_process_index},
        }

        assert isinstance(pretrained_model_name_or_path, str), "The `pretrained_model_path` should be a string."
        try:
            # If there is a trained peft adapter in the hub, load its config.
            remote_adapter_config = hf_hub_download(pretrained_model_name_or_path, "adapter_config.json")
        except:
            remote_adapter_config = None


        local_adapter_present =  os.path.exists(
            os.path.join(pretrained_model_name_or_path, "adapter_config.json")
        )

        # # Load the trained peft adapter config
        if local_adapter_present:
            trained_adapter_config = PeftConfig.from_pretrained(pretrained_model_name_or_path)
        else:
            trained_adapter_config = PeftConfig.from_pretrained(remote_adapter_config)

        # # Load the pretrained base model
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            trained_adapter_config.base_model_name_or_path,
            **pretrained_kwargs
        )

        # Load the Peft model by combing the base model with the trained adapter
        is_trainable = True # TODO: If following merge+train new adapter flow. Below should not be trainable!
        pretrained_model = PeftModel.from_pretrained(pretrained_model, pretrained_model_name_or_path, is_trainable=is_trainable)

        #pretrained_model.print_trainable_parameters()
        print("\nTrained peft adapter loaded for policy model\n")

        # Alternatively, load a peft model from a local path. See https://huggingface.co/docs/peft/quicktour. # TODO: DELETE. doesn't work
        # peft_model = AutoPeftModelForCausalLM.from_pretrained(pretrained_model_name_or_path)


        # Add value head to the pretrained peft model to create a policy network.
        if isinstance(pretrained_model, PeftModel):
            is_peft_model = True
        trl_model_args = {} # args for the value head
        # TODO: weights of v_head initialized using v_head_init_strategy="random" by default. trl also suports initialization using "norm".
        model = AutoModelForCausalLMWithValueHead(pretrained_model, **trl_model_args)
        # TODO: 1 VALUE HEAD REQURIES GRAD = FALSE AND NOT IN CUDA. CHECK IF BELOW CODE FIX THIS. 2. PEFTMODEL PRINT TRAINABLE PARAMETERS REUTRNS ... AND NONE
    

        # For back compatibility for class AutoModelForCausalLMWithValueHead. is_peft_model needs to be specified or calling model.state_dict() will fail.
        model.is_peft_model = is_peft_model
        # For back compatibility
        model.is_sequential_parallel = True
        model.current_device = Accelerator().local_process_index
        reward_adapter = None  # TODO: Consider adding reward adapter here? 
        if is_peft_model and reward_adapter is not None:
            model.add_and_load_reward_modeling_adapter(reward_adapter)
            model.supports_rm_adapter = True
        else:
            model.supports_rm_adapter = False


        # Adding v_head to device and register hook. See AutoModelForCausalLMWithValueHead.post_init(). 
        # TODO: is register_forward_hook necessary? outputs should be already on cuda
        first_device = list(set(model.pretrained_model.hf_device_map.values()))[0]
        model.v_head = model.v_head.to(first_device)
        def set_device_hook(module, input, outputs):
            new_output = ()
            for output in outputs:
                if isinstance(output, torch.Tensor):
                    new_output += (output.to(first_device),)
                else:
                    new_output += (output,)
            return new_output
        model.register_forward_hook(set_device_hook)
        self.base_model = model
        #breakpoint()
        # self.base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        #     rlhf_config.base_model_path,
        #     load_in_8bit=rlhf_config.load_in_8bit,
        #     device_map={"": Accelerator().local_process_index},
        #     peft_config=rlhf_config.lora_config_rl,
        # )
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.base_model,
            ref_model=None,
            tokenizer=self.base_tokenizer,
            dataset=self.base_dataset,
            data_collator=self.data_collator,
        )
        self.base_kwargs = {
            "top_k": rlhf_config.top_k,
            "top_p": rlhf_config.top_p,
            "do_sample": rlhf_config.do_sample,
            "pad_token_id": self.base_tokenizer.pad_token_id,
            "eos_token_id": rlhf_config.eos_token_id,
        }
        self.ppo_log_stats_dict = {}  # initialize the log stats dict

    def create_tokenizer(self, model_name):
        """
        Create a tokenizer for the given model name or model path.

        Args:
            model_name (str): The name of the model to create the tokenizer for.

        Returns:
            tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase): The tokenizer for the given model.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def data_collator(self, data):
        """
        Collate a batch of data samples into a dictionary.

        Args:
            data (List[Dict[str, Any]]): A list of data samples, where each sample is a dictionary.

        Returns:
            collated_data (Dict[str, List[Any]]): A dictionary where each key corresponds to a feature and the value is a list of values for that feature across all samples.
        """
        return dict((key, [d[key] for d in data]) for key in data[0])

    def create_dataset(self, tokenizer):
        """
        Build dataset for finetuning. This builds the dataset from `load_dataset`,
        one should customize this function to train the model on its own dataset.

        Args:
            tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase):
            The tokenizer to load a given dataset.

        Returns:
            dataset (datasets.Dataset): The dataset for finetuning.
        """
        args = self._rlhf_config
        if args.dataset_type == "local_db":
            qa_database = QuestionAnswerDatabase(
                db_file=self._rlhf_config.dataset_name
            )
            my_data_pd = qa_database.retrieve_all_question_answers_as_pandas()
            my_data_pd = my_data_pd[
                my_data_pd[QA_CSV_HEADER_VOTE_STATUS] == "up"
            ]
            my_data_pd = my_data_pd[
                [QA_CSV_HEADER_ID, QA_CSV_HEADER_QUESTION, QA_CSV_HEADER_ANSWER]
            ]
            print(
                "My local database has {} samples".format(my_data_pd.shape[0])
            )
            dataset = Dataset.from_dict(my_data_pd)
        elif args.dataset_type == "local_csv":  ## TODO: test
            dataset = load_dataset("csv", data_files=args.dataset_name)
            dataset = dataset[args.split]  # Convert DatasetDict to Dataset
        elif args.dataset_type == "huggingface":
            dataset = load_dataset(
                args.dataset_name,
                data_dir=args.dataset_subset_rl,
                split=args.split,
                use_auth_token=True,
                # num_proc=self.num_proc,
                # streaming=args.streaming,
            )
        else:
            raise FileNotFoundError(
                "No (supported) data files or dataset script found"
                f" {args.dataset_type}"
            )

        def preprocess_function(examples):
            """
            Preprocess a batch of examples for finetuning.

            Args:
                examples (Dict[str, Any]): A dictionary containing the examples to preprocess.

            Returns:
                new_examples (Dict[str, List[Any]]): A dictionary containing the preprocessed examples.
            """
            new_examples = {
                "query": [],
                "input_ids": [],
            }
            for question in examples[QA_CSV_HEADER_QUESTION]:
                query = "Question: " + question + "\n\nAnswer: "
                tokenized_question = tokenizer(query, truncation=True)
                new_examples["query"].append(query)
                new_examples["input_ids"].append(
                    tokenized_question["input_ids"]
                )
            return new_examples

        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
        )
        dataset = dataset.filter(
            lambda x: len(x["input_ids"]) < self._rlhf_config.max_seq_length,
            batched=False,
        )
        dataset.set_format(type="torch")

        return dataset

    def train(self, save_checkpoints_path=None):
        """
        Finetune the RL model using PPO algorithm.

        Args:
            save_checkpoints_path (str, optional): Path to save the model checkpoints.
            Default to None.
        """
        ## Initialize accelerator
        self.ppo_trainer.dataloader = self.accelerator.prepare(
            self.ppo_trainer.dataloader
        )
        ## training
        for epoch, batch in tqdm(enumerate(self.ppo_trainer.dataloader)):
            if epoch >= self._rlhf_config.total_epochs:
                break
            ## embed the questions and responses to tensors
            question_tensors = batch["input_ids"]
            response_tensors = self.ppo_trainer.generate(
                question_tensors,
                return_prompt=False,
                length_sampler=LengthSampler(
                    self._rlhf_config.output_min_length,
                    self._rlhf_config.output_max_length,
                ),
                **self.base_kwargs,
            )
            batch[QA_CSV_HEADER_ANSWER] = self.base_tokenizer.batch_decode(
                response_tensors, skip_special_tokens=True
            )
            # compute rewards and run PPO
            texts = [
                q + r
                for q, r in zip(batch["query"], batch[QA_CSV_HEADER_ANSWER])
            ]
            pipe_outputs = self.reward_pipe(texts, **self.reward_kwargs)
            rewards = [
                torch.tensor(
                    output[0]["score"] - self._rlhf_config.reward_baseline
                )
                for output in pipe_outputs
            ]
            stats = self.ppo_trainer.step(
                question_tensors, response_tensors, rewards
            )

            ## log stats
            self.log_stats_to_json(epoch=epoch, stats=stats, reward=rewards[0])
            # self.ppo_trainer.log_stats(stats, batch, rewards)
            print("\n\n\nstats: {}\n\n\n".format(stats))

            ## save weights
            if (
                self._rlhf_config.save_freq
                and epoch
                and epoch % self._rlhf_config.save_freq == 0
            ):
                if save_checkpoints_path is None:
                    save_checkpoints_path = os.path.join(
                        save_checkpoints_path,
                        "checkpoints",
                        f"checkpoints_epoch_{epoch}",
                    )
                self.ppo_trainer.save_pretrained(save_checkpoints_path)

    def log_stats_to_json(
        self, epoch, stats, reward, filename="ppo_log_stats.json"
    ):
        """
        Log the PPO stats to a json file.
        Args:
            epoch (int): The current epoch.
            stats (dict): The PPO stats.
            reward (float): The reward.
            filename (str, optional): The name of the json file. Defaults to "ppo_log_stats.json".
        """
        logs = self.ppo_log_stats_dict
        # Add new logs
        new_log = {}
        for stat_name, value in stats.items():
            if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                new_log[stat_name] = value.tolist()
            elif isinstance(value, (int, float, str, bool)) or value is None:
                new_log[stat_name] = value
            else:
                print(
                    f"Warning: Skipping non-serializable stat '{stat_name}' of"
                    f" type {type(value).__name__}"
                )
        new_log["reward"] = reward.tolist()
        # Write logs to file
        logs[f"epoch{epoch}"] = new_log
        with open(filename, "w") as json_file:
            json.dump(logs, json_file)
        # Update the class attribute
        self.ppo_log_stats_dict = logs

    def save(self, output_path=None):
        """
        Saves the RL findtuned model to the specified output path. If no output path is provided,
        the model is saved to the output directory specified in the RLHFConfig object.

        Args:
            output_path (str, optional): The path to save the model to. Default to None.
        """
        if output_path is None:
            output_path = os.path.join(
                self._rlhf_config.output_dir,
                self._rlhf_config.rl_lora_path,
            )
        self.ppo_trainer.save_pretrained(output_path)

    def train_and_save(self, output_path=None):
        """
        Finetune the model with RL and save it to the specified output path.
        If no output path is provided, the model is saved to the output directory
        specified in the RLHFConfig object.

        Args:
            output_path (str, optional): The path to save the model to. Default to None.
        """
        start_event = RLStartEvent(
            start_time=time.time(), date_time=datetime.utcfromtimestamp(time.time())
        )        
        self._telemetry.capture(start_event)
        self.train(save_checkpoints_path=output_path)
        self.save(output_path)
        self._telemetry.capture(
            RLStopEvent(
                end_time=time.time(),
                date_time=datetime.utcfromtimestamp(time.time()),
                duration=time.time() - start_event.start_time,
            )
        )
