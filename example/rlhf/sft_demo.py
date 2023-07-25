"""Demo for the supervised fine tuning."""

import pykoi

from datasets import Dataset
from pykoi.db.constants import (
    QA_CSV_HEADER_ID,
    QA_CSV_HEADER_QUESTION,
    QA_CSV_HEADER_ANSWER)

# get data from local database
qa_database = pykoi.QuestionAnswerDatabase()
my_data_pd = qa_database.retrieve_all_question_answers_as_pandas()
my_data_pd = my_data_pd[[QA_CSV_HEADER_ID,
                        QA_CSV_HEADER_QUESTION,
                        QA_CSV_HEADER_ANSWER]]

# analyze the data
print(my_data_pd)
print("My local database has {} samples".format(my_data_pd.shape[0]))
dataset = Dataset.from_dict(my_data_pd)

# run supervised finetuning
config = pykoi.RLHFConfig(base_model_path="meta-llama/Llama-2-7b-hf", dataset_type="local_db")
rlhf_step1_sft = pykoi.SFT(config)
rlhf_step1_sft.train_and_save("./models/rlhf_step1_sft")