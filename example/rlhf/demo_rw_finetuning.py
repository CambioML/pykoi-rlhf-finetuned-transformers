"""Demo for the reward fine tuning.

python -m example.rlhf.demo_rw_finetuning
"""

import pykoi

from pykoi.chat.db.constants import (
    RANKING_CSV_HEADER_ID,
    RANKING_CSV_HEADER_QUESTION,
    RANKING_CSV_HEADER_UP_RANKING_ANSWER,
    RANKING_CSV_HEADER_LOW_RANKING_ANSWER)

# get data from local database
ranking_database = pykoi.RankingDatabase()
my_data_pd = ranking_database.retrieve_all_question_answers_as_pandas()
my_data_pd = my_data_pd[[
    RANKING_CSV_HEADER_ID,
    RANKING_CSV_HEADER_QUESTION,
    RANKING_CSV_HEADER_UP_RANKING_ANSWER,
    RANKING_CSV_HEADER_LOW_RANKING_ANSWER]]

# analyze the data
print(my_data_pd)
print("My local database has {} samples in total".format(my_data_pd.shape[0]))

# run reward model finetuning
# config = pykoi.RLHFConfig(dataset_type="local_db")
config = pykoi.RLHFConfig(reward_model_path = "databricks/dolly-v2-3b")
rlhf_step2_rft = pykoi.RewardFinetuning(config)
rlhf_step2_rft.train_and_save("./models/rlhf_step2_rw")
