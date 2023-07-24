"""Demo for the supervised fine tuning."""

from datasets import Dataset


import pykoi.cambio as cb


QA_CSV_HEADER_ID = 'ID'
QA_CSV_HEADER_QUESTION = 'Question'
QA_CSV_HEADER_ANSWER = 'Answer'
QA_CSV_HEADER_VOTE_STATUS = 'Vote Status'
QA_CSV_HEADER_TIMESTAMPS = 'Timestamp'
QA_CSV_HEADER = (
    QA_CSV_HEADER_ID,
    QA_CSV_HEADER_QUESTION,
    QA_CSV_HEADER_ANSWER,
    QA_CSV_HEADER_VOTE_STATUS,
    QA_CSV_HEADER_TIMESTAMPS
)

qa_database = cb.QuestionAnswerDatabase()

my_data_pd = qa_database.retrieve_all_question_answers_as_pandas()
print(my_data_pd)

my_data_pd = my_data_pd[[QA_CSV_HEADER_ID,
                        QA_CSV_HEADER_QUESTION,
                        QA_CSV_HEADER_ANSWER]]
print(my_data_pd)
print("My local database has {} samples".format(my_data_pd.shape[0]))
dataset = Dataset.from_dict(my_data_pd)
dataset


config = cb.RLHFConfig(base_model_path="meta-llama/Llama-2-7b-hf", dataset_type="local_db")

rlhf_step1_sft = cb.SFT(config)

rlhf_step1_sft.train_and_save("./models/rlhf_step1_sft")
