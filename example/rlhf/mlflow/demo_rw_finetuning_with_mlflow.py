"""Demo for the reward fine tuning.

python -m example.rlhf.demo_rw_finetuning_with_mlfow
"""

from pykoi.rlhf import RLHFConfig
from pykoi.rlhf import RewardFinetuning
from pykoi.chat import RankingDatabase
import mlflow
import datetime

from pykoi.chat.db.constants import (
    RANKING_CSV_HEADER_ID,
    RANKING_CSV_HEADER_QUESTION,
    RANKING_CSV_HEADER_UP_RANKING_ANSWER,
    RANKING_CSV_HEADER_LOW_RANKING_ANSWER)

# get data from local database
ranking_database = RankingDatabase()
my_data_pd = ranking_database.retrieve_all_question_answers_as_pandas()
my_data_pd = my_data_pd[[
    RANKING_CSV_HEADER_ID,
    RANKING_CSV_HEADER_QUESTION,
    RANKING_CSV_HEADER_UP_RANKING_ANSWER,
    RANKING_CSV_HEADER_LOW_RANKING_ANSWER]]

# analyze the data
print(my_data_pd)
print("My local database has {} samples in total".format(my_data_pd.shape[0]))

# Set up mlflow experiment name.
# mlflow.set_tracking_uri("http://x.x.x.x:5000")
experiment = "rlhf_step2_rw"
current_time = str(datetime.datetime.now())
mlflow_experiment_name = '/'.join([experiment, current_time])
mlflow.set_experiment(mlflow_experiment_name)

# Set pykoi parameters.
reward_model_path = "databricks/dolly-v2-3b"
trained_model_path = "./models/rlhf_step2_rw"

# Manually log pykoi parameters into mlflow. Torch level parameters are automatically logged.
mlflow.log_param("pykoi_reward_model_path", reward_model_path)
mlflow.log_param("pykoi_trained_model_path", trained_model_path)

# run reward model finetuning
# config = RLHFConfig(dataset_type="local_db")
config = RLHFConfig(reward_model_path=reward_model_path)
rlhf_step2_rft = RewardFinetuning(config)
rlhf_step2_rft.train_and_save(trained_model_path)

# Save the trained reward model into mlflow artifacts.
mlflow.log_artifacts(trained_model_path)
