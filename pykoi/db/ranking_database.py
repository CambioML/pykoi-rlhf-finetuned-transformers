"""This file contains the RankingDatabase class which is used to store and retrieve"""
import csv
import os
import sqlite3
import threading

import pandas as pd

from pykoi.db.constants import RANKING_CSV_HEADER


class RankingDatabase:
    """Ranking Database class"""

    def __init__(
        self,
        db_file: str = os.path.join(os.getcwd(), "ranking.db"),
        debug: bool = False,
    ):
        """
        Initializes a new instance of the RankingDatabase class.

        Args:
            db_file (str): The path to the SQLite database file.
            debug (bool, optional): Whether to print debug messages. Defaults to False.
        """
        self._db_file = db_file
        self._debug = debug
        self._local = threading.local()  # Thread-local storage
        self._lock = threading.Lock()  # Lock for concurrent write operations
        self.create_table()

    def get_connection(self):
        """Returns the thread-local database connection"""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(self._db_file)
        return self._local.connection

    def get_cursor(self):
        """Returns the thread-local database cursor"""
        if not hasattr(self._local, "cursor"):
            self._local.cursor = self.get_connection().cursor()
        return self._local.cursor

    def create_table(self):
        """
        Creates the ranking table if it does not already exist in the database.
        The table has four columns: id (primary key), question, up_ranking_answer, and low_ranking_answer.
        """
        query = """
        CREATE TABLE IF NOT EXISTS ranking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            up_ranking_answer TEXT,
            low_ranking_answer TEXT
        );
        """
        with self._lock:
            cursor = self.get_cursor()
            cursor.execute(query)
            self.get_connection().commit()

        if self._debug:
            rows = self.retrieve_all_question_answers()
            print("Table contents after creating table:")
            self.print_table(rows)

    def insert_ranking(
        self, question: str, up_ranking_answer: str, low_ranking_answer: str
    ):
        """
        Inserts a new ranking entry into the database with the given question, up_ranking_answer,
        and low_ranking_answer.

        Args:
            question (str): The question of the ranking entry.
            up_ranking_answer (str): The higher ranked answer of the ranking entry.
            low_ranking_answer (str): The lower ranked answer of the ranking entry.

        Returns:
            int: The ID of the newly inserted row.
        """
        query = """
        INSERT INTO ranking (question, up_ranking_answer, low_ranking_answer)
        VALUES (?, ?, ?);
        """
        with self._lock:
            cursor = self.get_cursor()
            cursor.execute(query, (question, up_ranking_answer, low_ranking_answer))
            self.get_connection().commit()

        if self._debug:
            rows = self.retrieve_all_question_answers()
            print("Table contents after inserting entry:")
            self.print_table(rows)

        return cursor.lastrowid

    def retrieve_all_question_answers(self):
        """
        Retrieves all question-answer pairs from the database.

        Returns:
            list: A list of tuples representing the question-answer pairs.
        """
        query = """
        SELECT * FROM ranking;
        """
        with self._lock:
            cursor = self.get_cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
        return rows

    def close_connection(self):
        """
        Closes the connection to the database.
        """
        if hasattr(self._local, "cursor"):
            self._local.cursor.close()
            del self._local.cursor
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection

    def print_table(self, rows):
        """
        Prints the contents of the table in a formatted manner.

        Args:
            rows (list): A list of tuples where each tuple represents a row in the table.
                         Each tuple contains four elements: ID, Question, Up_Ranking_Answer, Low_Ranking_Answer.
        """
        for row in rows:
            print(
                f"ID: {row[0]}, Question: {row[1]}, "
                f"Up_Ranking_Answer: {row[2]}, Low_Ranking_Answer: {row[3]}"
            )

    def save_to_csv(self, csv_file_name="ranking_data.csv"):
        """
        Saves the contents of the ranking table into a CSV file.

        Args:
            csv_file_name (str, optional): The name of the CSV file to which the data will be written.
            Defaults to "ranking_data.csv".

        The CSV file will have the following columns: ID, Question, Up_Ranking_Answer, Low_Ranking_Answer.
        Each row in the CSV file corresponds to a row in the ranking table.

        This method retrieves all ranking entries from the database by calling the
        retrieve_all_question_answers method. It then writes this data to the CSV file.
        """
        ranking_data = self.retrieve_all_question_answers()

        with open(csv_file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(RANKING_CSV_HEADER)
            writer.writerows(ranking_data)

    def retrieve_all_question_answers_as_pandas(self):
        """
        Retrieves pairs from the database as a pandas dataframe.

        Returns:
            DataFrame: A pandas dataframe.
        """
        rows = self.retrieve_all_question_answers()
        rows_to_pd = pd.DataFrame(rows)
        rows_to_pd.columns = RANKING_CSV_HEADER
        return rows_to_pd
