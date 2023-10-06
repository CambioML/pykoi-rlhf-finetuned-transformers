"""Demo for the supervised fine tuning.

python -m example.rlhf.supervised_finetuning_demo
"""

from pykoi.chat import QuestionAnswerDatabase
from pykoi.rlhf import RLHFConfig
from pykoi.rlhf import SupervisedFinetuning
import mlflow
import datetime

from pykoi.chat.db.constants import (
    QA_CSV_HEADER_ID,
    QA_CSV_HEADER_QUESTION,
    QA_CSV_HEADER_ANSWER,
    QA_CSV_HEADER_VOTE_STATUS)

# get data from local database
qa_database = QuestionAnswerDatabase()
my_data_pd = qa_database.retrieve_all_question_answers_as_pandas()
my_data_pd = my_data_pd[[
    QA_CSV_HEADER_ID,
    QA_CSV_HEADER_QUESTION,
    QA_CSV_HEADER_ANSWER,
    QA_CSV_HEADER_VOTE_STATUS]]

# analyze the data
print(my_data_pd)
print("My local database has {} samples in total".format(my_data_pd.shape[0]))

# Set up mlflow experiment
# mlflow.set_tracking_uri("http://x.x.x.x:5000")
mlflow.set_experiment("rlhf_step1_sft/" + str(datetime.datetime.now()))

# Set pykoi parameters
base_model_path = "databricks/dolly-v2-3b"
dataset_type = "local_db"
peft_model_path = "./models/rlhf_step1_sft"

# Manually log pykoi parameters into mlflow. Other parameters at torch level are automatically logged.
mlflow.log_param("pykoi_base_model_path", base_model_path)
mlflow.log_param("pykoi_dataset_type", dataset_type)
mlflow.log_param("pykoi_peft_model_path", peft_model_path)

# Run supervised finetuning.
# Training metrics are automatically logged.
config = RLHFConfig(base_model_path=base_model_path, dataset_type=dataset_type)
rlhf_step1_sft = SupervisedFinetuning(config)
rlhf_step1_sft.train_and_save(peft_model_path)

# Save the trained peft model into mlflow artifacts.
mlflow.log_artifacts(peft_model_path)
