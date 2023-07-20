"""This file contains the RankingDatabase class which is used to store and retrieve"""
import csv
import os
import sqlite3
import threading


CSV_HEADER = ('ID', 'Question', 'Answer', 'Ranking')


class RankingDatabase:
    """Ranking Database class"""

    def __init__(self,
                 db_file: str = os.path.join(os.getcwd(), 'ranking.db'),
                 debug: bool = False):
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
        The table has four columns: id (primary key), question, answer, and ranking.
        The ranking column is constrained to only allow the values 1 and 2.
        """
        query = """
        CREATE TABLE IF NOT EXISTS ranking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            ranking INTEGER CHECK (ranking IN (1, 2))
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

    def insert_ranking(self, question: str, answer: str, ranking: int):
        """
        Inserts a new ranking entry into the database with the given question, answer, and ranking.

        Args:
            question (str): The question of the ranking entry.
            answer (str): The answer of the ranking entry.
            ranking (int): The ranking value, should be either 1 or 2.

        Returns:
            int: The ID of the newly inserted row.
        """
        query = """
        INSERT INTO ranking (question, answer, ranking)
        VALUES (?, ?, ?);
        """
        with self._lock:
            cursor = self.get_cursor()
            cursor.execute(query, (question, answer, ranking))
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
                         Each tuple contains four elements: ID, Question, Answer, and Ranking.
        """
        for row in rows:
            print(f"ID: {row[0]}, Question: {row[1]}, "
                  f"Answer: {row[2]}, Ranking: {row[3]}")

    def save_to_csv(self, csv_file_name="ranking_data.csv"):
        """
        Saves the contents of the ranking table into a CSV file.

        Args:
            csv_file_name (str, optional): The name of the CSV file to which the data will be written.
            Defaults to "ranking_data.csv".

        The CSV file will have the following columns: ID, Question, Answer, Ranking. Each row in the
        CSV file corresponds to a row in the ranking table.

        This method retrieves all ranking entries from the database by calling the
        retrieve_all_question_answers method. It then writes this data to the CSV file.
        """
        ranking_data = self.retrieve_all_question_answers()

        with open(csv_file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADER)
            writer.writerows(ranking_data)
