"""Question answer database module"""
import csv
import datetime
import os
import sqlite3
import threading

import pandas as pd

from pykoi.chat.db.constants import QA_CSV_HEADER, QA_LIST_SEPARATOR


class RAGDatabase:
    """RAG Question Answer Database class"""

    def __init__(
        self,
        db_file: str = os.path.join(os.getcwd(), "rag.db"),
        debug: bool = False,
    ):
        """
        Initializes a new instance of the RAGDatabase class.

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
        Creates the question_answer table if it does not already exist in the database.
        The table has 6 columns: id (primary key), question, answer, source, source_content, and vote_status.
        vote_status is a text field that can only have the values 'up', 'down', or 'n/a'.
        """
        query = """
        CREATE TABLE IF NOT EXISTS rag_question_answer (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            edited_answer TEXT,
            vote_status TEXT CHECK (vote_status IN ('up', 'down', 'n/a')),
            rag_sources TEXT,
            source TEXT,
            source_content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

    def insert_question_answer(self, question: str, answer: str, rag_sources: list, source: list, source_content: list):
        """
        Inserts a new question-answer pair into the database with the given question and answer.
        The vote_status field is set to 'n/a' by default.
        Returns the ID of the newly inserted row.

        Args:
            question (str): The question to insert.
            answer (str): The answer to insert.
            rag_sources (list): The RAG sources of the question-answer pair.
            source (list): The source(s) of the question-answer pair.
            source_content (list): The source content(s) of the question-answer pair.

        Returns:
            int: The ID of the newly inserted row.
        """
        timestamp = datetime.datetime.now()
        query = """
        INSERT INTO rag_question_answer (question, answer, edited_answer, rag_sources, source, source_content, vote_status, timestamp)
        VALUES (?, ?, '', ?, ?, ?, 'n/a', ?);
        """
        rag_sources = QA_LIST_SEPARATOR.join(rag_sources)
        source = QA_LIST_SEPARATOR.join(source)
        source_content = QA_LIST_SEPARATOR.join(source_content)
        print("rag insert question answer", rag_sources)

        with self._lock:
            cursor = self.get_cursor()
            cursor.execute(query, (question, answer, rag_sources, source, source_content, timestamp))
            self.get_connection().commit()

        if self._debug:
            rows = self.retrieve_all_question_answers()
            print("Table contents after inserting table:")
            self.print_table(rows)

        return cursor.lastrowid

    def update_vote_status(self, id, vote_status):
        """
        Updates the vote status of a question-answer pair with the given ID.

        Args:
            id (int): The ID of the question-answer pair to update.
            vote_status (str): The new vote status to set. Must be one of 'up', 'down', or 'n/a'.

        Raises:
            ValueError: If the question with the given ID does not exist.
        """
        query = """
        UPDATE rag_question_answer
        SET vote_status = ?
        WHERE id = ?;
        """
        with self._lock:
            cursor = self.get_cursor()
            cursor.execute(query, (vote_status, id))
            self.get_connection().commit()

        if cursor.rowcount == 0:
            raise ValueError(f"Question with ID {id} does not exist.")

        if self._debug:
            rows = self.retrieve_all_question_answers()
            print("Table contents after updating table:")
            self.print_table(rows)

    def update_answer(self, id, new_answer):
        """
        Updates the edited_answer of a question-answer pair with the given ID.

        Args:
            id (int): The ID of the question-answer pair to update.
            new_answer (str): The new modified response to set.

        Raises:
            ValueError: If the question with the given ID does not exist.
        """
        query = """
        UPDATE rag_question_answer
        SET edited_answer = ?
        WHERE id = ?;
        """
        print('update_answer',new_answer)
        with self._lock:
            cursor = self.get_cursor()
            cursor.execute(query, (new_answer, id))
            self.get_connection().commit()

        if cursor.rowcount == 0:
            raise ValueError(f"Question with ID {id} does not exist.")

        if self._debug:
            rows = self.retrieve_all_question_answers()
            print("Table contents after updating table:")
            self.print_table(rows)

    def retrieve_all_question_answers(self):
        """
        Retrieves all question-answer pairs from the database.

        Returns:
            list: A list of tuples representing the question-answer pairs.
        """
        query = """
        SELECT * FROM rag_question_answer;
        """
        with self._lock:
            cursor = self.get_cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
        return rows

    def retrieve_all_question_answers_as_pandas(self):
        """
        Retrieves all question-answer pairs from the database as a pandas dataframe.

        Returns:
            DataFrame: A pandas dataframe.
        """
        rows = self.retrieve_all_question_answers()
        rows_to_pd = pd.DataFrame(rows)
        rows_to_pd.columns = QA_CSV_HEADER
        return rows_to_pd

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
                        Each tuple contains five elements: ID, Question, Answer, Timestamp, and Vote Status.
        """
        for row in rows:
            print(
                f"ID: {row[0]}, Question: {row[1]}, "
                f"Answer: {row[2]}, Vote Status: {row[3]}, Timestamp: {row[4]}"
            )

    def save_to_csv(self, csv_file_name="question_answer_votes.csv"):
        """
        This method saves the contents of the question_answer table into a CSV file.

        Args:
            csv_file_name (str, optional): The name of the CSV file to which the data will be written.
            Defaults to "question_answer_votes.csv".

        The CSV file will have the following columns: ID, Question, Answer, Vote Status. Each row in the
        CSV file corresponds to a row in the question_answer table.

        This method first retrieves all question-answer pairs from the database by calling the
        retrieve_all_question_answers method. It then writes this data to the CSV file.
        """
        my_sql_data = self.retrieve_all_question_answers()

        with open(csv_file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(QA_CSV_HEADER)
            writer.writerows(my_sql_data)
