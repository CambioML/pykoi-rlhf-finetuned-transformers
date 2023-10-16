"""Demo for using uniflow to generate data for supervised fine tuning.

python -m example.uniflow.uniflow_sft_demo
"""
import os
import pandas as pd

from uniflow.flow.flow import Flow
from pykoi.rlhf import RLHFConfig
from pykoi.rlhf import SupervisedFinetuning
from pykoi.chat.db.constants import (
    QA_CSV_HEADER_ID,
    QA_CSV_HEADER_QUESTION,
    QA_CSV_HEADER_ANSWER,
    QA_CSV_HEADER_VOTE_STATUS)

CSV_FILENAME = "qd_immigration"
CSV_OUTPUT_SUFFIX = "-flow-output"

# Load data
current_directory = os.getcwd()
qaa = pd.read_csv(f"{current_directory}/{CSV_FILENAME}.csv", encoding="utf8")

# run flow
flow = Flow()
output_dict = flow(qaa)

# save new data to csv
df = pd.DataFrame(output_dict["output"][0], columns=[
    QA_CSV_HEADER_ID,
    QA_CSV_HEADER_QUESTION,
    QA_CSV_HEADER_ANSWER,
    QA_CSV_HEADER_VOTE_STATUS])
df.to_csv(f"{current_directory}/{CSV_FILENAME}{CSV_OUTPUT_SUFFIX}.csv", index=False)

# analyze the data
print("Flow save successful!")
print(df)
print(f"The output csv file {CSV_FILENAME}{CSV_OUTPUT_SUFFIX}.csv has {df.shape[0]} rows in total")

# run supervised finetuning
config = RLHFConfig(base_model_path="databricks/dolly-v2-3b", dataset_type="local_csv", dataset_name=f"{CSV_FILENAME}{CSV_OUTPUT_SUFFIX}.csv")
rlhf_step1_sft = SupervisedFinetuning(config)
rlhf_step1_sft.train_and_save("./models/rlhf_step1_sft")
