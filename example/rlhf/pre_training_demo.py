"""Demo for the pre-training.

python -m example.rlhf.pre_training_demo
"""

from pykoi.chat import QuestionAnswerDatabase
from pykoi.chat.db.constants import (QA_CSV_HEADER_ANSWER, QA_CSV_HEADER_ID,
                                     QA_CSV_HEADER_QUESTION,
                                     QA_CSV_HEADER_VOTE_STATUS)
from pykoi.rlhf import RLHFConfig, PreTraining

# get data from local database
qa_database = QuestionAnswerDatabase()
my_data_pd = qa_database.retrieve_all_question_answers_as_pandas()
my_data_pd = my_data_pd[
    [
        QA_CSV_HEADER_ID,
        QA_CSV_HEADER_QUESTION,
        QA_CSV_HEADER_ANSWER,
        QA_CSV_HEADER_VOTE_STATUS,
    ]
]

# analyze the data
print(my_data_pd)
print("My local database has {} samples in total".format(my_data_pd.shape[0]))

# run pre-training
config = RLHFConfig(base_model_path="databricks/dolly-v2-3b", dataset_type="local_db")
rlhf_step1_sft = PreTraining(config)
rlhf_step1_sft.train_and_save("./models/rlhf_step1_sft")
