"""Demo for the supervised fine tuning."""

import pykoi

from pykoi.chat.db.constants import (
    QA_CSV_HEADER_ID,
    QA_CSV_HEADER_QUESTION,
    QA_CSV_HEADER_ANSWER,
    QA_CSV_HEADER_VOTE_STATUS)

# get data from local database
qa_database = pykoi.QuestionAnswerDatabase()
my_data_pd = qa_database.retrieve_all_question_answers_as_pandas()
my_data_pd = my_data_pd[[
    QA_CSV_HEADER_ID,
    QA_CSV_HEADER_QUESTION,
    QA_CSV_HEADER_ANSWER,
    QA_CSV_HEADER_VOTE_STATUS]]

# analyze the data
print(my_data_pd)
print("My local database has {} samples in total".format(my_data_pd.shape[0]))

# run supervised finetuning
config = pykoi.RLHFConfig(base_model_path="elinas/llama-7b-hf-transformers-4.29", dataset_type="local_db")
rlhf_step1_sft = pykoi.SupervisedFinetuning(config)
rlhf_step1_sft.train_and_save("./models/rlhf_step1_sft")
