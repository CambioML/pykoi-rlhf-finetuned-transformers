"""Chatbot comparator component."""
import time
import pandas as pd

from typing import List

from pykoi.component.base import Component
from pykoi.db.comparator_database import ComparatorDatabase, ComparatorQuestionDatabase
from pykoi.llm.abs_llm import AbsLlm
from pykoi.interactives.barchart import Barchart


def df_to_js_array(df):
    records = df.to_dict(orient="records")
    return records


class Compare(Component):
    """Chatbot comparator component."""

    def __init__(self, models: List[AbsLlm], **kwargs):
        """
        Initializes a new instance of the Compare class.

        Args:
            models (List[AbsLlm]): A list of models to compare.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            Exception: If a model with the same name already exists.

        """
        super().__init__(None, "Compare", **kwargs)
        self.models = {}
        for model in models:
            self.add(model)
        self.question_db = ComparatorQuestionDatabase()
        self.comparator_db = ComparatorDatabase()
        self.inference_results = pd.DataFrame(
            columns=["question", "model", "answer", "latency", "length_in_tokens"]
        )  # Added this line

    def add(self, model: AbsLlm):
        """
        Adds a model to the comparator.

        Args:
            model (AbsLlm): The model to add.

        Raises:
            Exception: If a model with the same name already exists.
        """
        if model.name in self.models:
            raise Exception(f"Model {model.name} already exists")
        self.models[model.name] = model

    def inference(self, questions: List[str]):
        """
        Run each model on each question and store the results in a pandas dataframe.

        Args:
            questions (List[str]): The list of questions.

        Returns:
            pd.DataFrame: DataFrame containing the questions, models, answers, latency, and answer length.
        """
        # Initialize an empty list to store the results
        results = []

        # Iterate through each question
        for question in questions:
            # Iterate through each model
            for model_name, model in self.models.items():
                # Record the start time
                start_time = time.time()

                # Get the response from the model
                response = model.predict(question, num_of_response=1)

                # Calculate the elapsed time
                latency = time.time() - start_time

                # Calculate the length of the response in tokens (assuming white space as token separator)
                length_in_tokens = len(response[0].split())

                # Store the question, model name, response, latency, and length in the results list
                results.append(
                    [question, model_name, response[0], latency, length_in_tokens]
                )

        # Convert the results into a DataFrame
        self.inference_results = pd.DataFrame(
            results,
            columns=["question", "model", "answer", "latency", "length_in_tokens"],
        )

        return self.inference_results

    def visualize(self):
        records = df_to_js_array(self.inference_results)
        return Barchart()(data=records)
