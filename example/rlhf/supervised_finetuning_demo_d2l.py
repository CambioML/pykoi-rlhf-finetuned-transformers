"""Demo for the supervised fine tuning.

python -m example.rlhf.supervised_finetuning_demo_d2l
"""

from peft import LoraConfig
from pykoi.chat import QuestionAnswerDatabase
from pykoi.chat.db.constants import (QA_CSV_HEADER_ANSWER, QA_CSV_HEADER_ID,
                                     QA_CSV_HEADER_QUESTION,
                                     QA_CSV_HEADER_VOTE_STATUS)
from pykoi.rlhf import RLHFConfig, SupervisedFinetuning
from trl import DataCollatorForCompletionOnlyLM



# run supervised finetuning
config = RLHFConfig(base_model_path="mistralai/Mistral-7B-Instruct-v0.1",
                    dataset_type="local_csv", dataset_name="data/chapter22_trnvalfromseed_data_processed.csv",
                    train_test_split_ratio=0,  # ratio for test set DH:TODO: COBINE TRAIN AND EVAL
                    max_seq_length=896,
                    per_device_eval_batch_size=1,
                    log_freq=20,
                    # dh: NOTE: 1 EPOCH iterates the dataset once. So log freq 20 means iterating 20 entries when training batch size = 1.
                    # (i.e., log_freq = 0.12 epoch when the dataset has 166 entires).
                    save_freq=40000,
                    num_train_epochs=20,
                    max_steps=-1,  # if a positive number is given, it will override num_train_epochs
                    device_map="auto",
                    lora_config_rl=LoraConfig(
                        r=512,
                        lora_alpha=1024,
                        lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", ],  # "gate_proj","up_proj","down_proj",], #"lm_head",],
                        bias="none",
                        task_type="CAUSAL_LM"
                    ),
                    data_collator=DataCollatorForCompletionOnlyLM,
                    no_evaluation=True,
                    prepare_text="d2l",
                    split = "train[:10%]"
                    )
rlhf_step1_sft = SupervisedFinetuning(config)
rlhf_step1_sft.train_and_save("./models/rlhf_step1_sft")
