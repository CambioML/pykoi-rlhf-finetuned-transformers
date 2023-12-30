"""Comparator Database"""
import csv
import datetime
import os
from typing import List, Tuple

import pandas as pd

from pykoi.chat.db.abs_database import AbsDatabase
from pykoi.chat.db.constants import COMPARATOR_CSV_HEADER


class ComparatorQuestionDatabase(AbsDatabase):
    """Comparator Question Database class"""

    def __init__(
        self,
        db_file: str = os.path.join(os.getcwd(), "comparator.db"),
        debug: bool = False,
    ) -> None:
        """
        Initializes a new instance of the ComparatorQuestionDatabase class.

        Args:
            db_file (str): The path to the SQLite database file.
            debug (bool, optional): Whether to print debug messages. Defaults to False.
        """
        query = """
        CREATE TABLE IF NOT EXISTS comparator_question (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        super().__init__(db_file, debug)
        self.create_table(query)

    def insert(self, **kwargs) -> None:
        """
        Inserts question, timestamp into the database.

        Args:
            kwargs (dict): The key-value pairs to insert into the database.

        Returns:
            int: The ID of the newly inserted row.
        """
        question = kwargs["question"]
        timestamp = datetime.datetime.now()
        query = """
        INSERT INTO comparator_question (question, timestamp)
        VALUES (?, ?);
        """
        with self._lock:
            cursor = self.get_cursor()
            cursor.execute(query, (question, timestamp))
            self.get_connection().commit()

        if self._debug:
            rows = self.retrieve_all()
            print("Table contents after inserting table:")
            self.print_table(rows)

        return cursor.lastrowid

    def update(self, **kwargs) -> None:
        """
        Updates the database.
        """
        raise NotImplementedError("ComparatorQuestionDatabase does not support update.")

    def retrieve_all(self) -> List[Tuple]:
        """
        Retrieves all pairs from the database.

        Returns:
            list: A list of tuples.
        """
        query = """
        SELECT * FROM comparator_question;
        """
        with self._lock:
            cursor = self.get_cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
        return rows

    def print_table(self, rows: List[Tuple]) -> None:
        """
        Prints the contents of the table in a formatted manner.

        Args:
            rows (list): A list of tuples where each tuple represents a row in the table.
                        Each tuple contains five elements: ID, Question.
        """
        for row in rows:
            print(f"ID: {row[0]}, Question: {row[1]}, Timestamp: {row[2]}")


class ComparatorDatabase(AbsDatabase):
    """ComparatorDatabase class."""

    def __init__(
        self,
        db_file: str = os.path.join(os.getcwd(), "comparator.db"),
        debug: bool = False,
    ) -> None:
        query = """
        CREATE TABLE IF NOT EXISTS comparator (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            qid INTEGER NOT NULL,
            rank INTEGER NOT NULL,
            answer TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        super().__init__(db_file, debug)
        self.create_table(query)

    def insert(self, **kwargs) -> None:
        """
        Inserts a new row into the comparator table.

        Args:
            kwargs (dict): The key-value pairs to insert into the database.
        """
        timestamp = datetime.datetime.now()
        check_query = """
        SELECT * FROM comparator
        WHERE model = ? AND qid = ?;
        """
        with self._lock:
            cursor = self.get_cursor()
            cursor.execute(check_query, (kwargs["model"], kwargs["qid"]))
            existing_row = cursor.fetchone()

            if existing_row is not None:
                raise ValueError(
                    f"Row with model={kwargs['model']} and"
                    f" qid={kwargs['qid']} already exists"
                )

            query = """
            INSERT INTO comparator (model, qid, rank, answer, timestamp)
            VALUES (?, ?, ?, ?, ?);
            """
            cursor.execute(
                query,
                (
                    kwargs["model"],
                    kwargs["qid"],
                    kwargs["rank"],
                    kwargs["answer"],
                    timestamp,
                ),
            )
            self.get_connection().commit()

        if self._debug:
            rows = self.retrieve_all()
            print("Table contents after inserting table")
            self.print_table(rows)

    def update(self, **kwargs) -> None:
        """
        Updates the rank of a row in the comparator table by its id.

        Args:
            kwargs (dict): The key-value pairs to update in the database.
        """
        query = """
        UPDATE comparator
        SET rank = ?
        WHERE qid = ? AND model = ?;
        """
        with self._lock:
            cursor = self.get_cursor()
            cursor.execute(query, (kwargs["rank"], kwargs["qid"], kwargs["model"]))
            self.get_connection().commit()
        if self._debug:
            rows = self.retrieve_all()
            print("Table contents after updating table")
            self.print_table(rows)

    def retrieve_all(self) -> List[Tuple]:
        """
        Retrieves all pairs from the database.

        Returns:
            list: A list of tuples.
        """
        query = """
        SELECT * FROM comparator;
        """
        with self._lock:
            cursor = self.get_cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
        return rows

    def retrieve_all_question_answers(self):
        """
        Retrieves all question-answer pairs from the database.

        Returns:
            rows: rows of data of the question-answer pairs.
        """
        query = """
        SELECT comparator.id, comparator.model, comparator.qid, comparator_question.question, comparator.answer, comparator.rank, comparator.timestamp
        FROM comparator
        JOIN comparator_question
        ON comparator.qid = comparator_question.id;
        """
        with self._lock:
            cursor = self.get_cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
        return rows

    def print_table(self, rows: List[Tuple]) -> None:
        """
        Prints the comparator table.

        Args:
            rows (list): A list of tuples where each tuple represents a row in the table.
                        Each tuple contains five elements: ID, Model, QID, Rank, Answer, Timestamp.
        """
        for row in rows:
            print(
                f"ID: {row[0]}, "
                f"Model: {row[1]}, "
                f"QID: {row[2]}, "
                f"Rank: {row[3]}, "
                f"Answer: {row[4]}, "
                f"Timestamp: {row[5]}"
            )

    def save_to_csv(self, csv_file_name="comparator_table"):
        """
        This method saves the contents of the RAG table into a CSV file.

        Args:
            csv_file_name (str, optional): The name of the CSV file to which the data will be written.
            Defaults to "comparator_table".

        The CSV file will have the following columns: TODO. Each row in the
        CSV file corresponds to a row in the question_answer table.

        This method first retrieves all question-answer pairs from the database by calling the
        retrieve_all method. It then writes this data to the CSV file.
        """

        my_sql_data = self.retrieve_all_question_answers()

        with open(csv_file_name + ".csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(COMPARATOR_CSV_HEADER)
            writer.writerows(my_sql_data)

    def retrieve_all_question_answers_as_pandas(self) -> pd.DataFrame:
        """
        Retrieves all data by joining the comparator and comparator_question tables as a pandas dataframe.

        Returns:
            DataFrame: A pandas dataframe.
        """
        join_query = """
        SELECT 
            comparator.id, 
            comparator.model, 
            comparator.qid, 
            comparator_question.question, 
            comparator.rank, 
            comparator.answer, 
            comparator.timestamp
        FROM comparator
        INNER JOIN comparator_question 
        ON comparator.qid = comparator_question.id;
        """

        with self._lock:
            cursor = self.get_cursor()
            cursor.execute(join_query)
            rows = cursor.fetchall()

        df = pd.DataFrame(
            rows,
            columns=["ID", "Model", "QID", "Question", "Rank", "Answer", "Timestamp"],
        )
        return df
