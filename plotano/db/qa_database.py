"""Question answer database module"""
import sqlite3


class QuestionAnswerDatabase:
    """Question Answer Database class"""

    def __init__(self, db_file: str):
        """
        Initializes a new instance of the QuestionAnswerDatabase class.

        Args:
            db_file (str): The path to the SQLite database file.
        """
        self._conn = sqlite3.connect(db_file)
        self.create_table()

    def create_table(self):
        """
        Creates the question_answer table if it does not already exist in the database.
        The table has four columns: id (primary key), question, answer, and vote_status.
        vote_status is a text field that can only have the values 'up', 'down', or 'n/a'.
        """
        query = """
        CREATE TABLE IF NOT EXISTS question_answer (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            vote_status TEXT CHECK (vote_status IN ('up', 'down', 'n/a'))
        );
        """
        self._conn.execute(query)
        self._conn.commit()

    def insert_question_answer(self, question: str, answer: str):
        """
        Inserts a new question-answer pair into the database with the given question and answer.
        The vote_status field is set to 'n/a' by default.
        Returns the ID of the newly inserted row.

        Args:
            question (str): The question to insert.
            answer (str): The answer to insert.

        Returns:
            int: The ID of the newly inserted row.
        """
        query = """
        INSERT INTO question_answer (question, answer, vote_status)
        VALUES (?, ?, 'n/a');
        """
        cursor = self._conn.cursor()
        cursor.execute(query, (question, answer))
        self._conn.commit()

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
        UPDATE question_answer
        SET vote_status = ?
        WHERE id = ?;
        """
        cursor = self._conn.cursor()
        cursor.execute(query, (vote_status, id))
        self._conn.commit()

        if cursor.rowcount == 0:
            raise ValueError(f"Question with ID {id} does not exist.")

    def retrieve_all_question_answers(self):
        """
        Retrieves all question-answer pairs from the database.

        Returns:
            list: A list of tuples representing the question-answer pairs.
        """
        query = """
        SELECT * FROM question_answer;
        """
        cursor = self._conn.execute(query)
        rows = cursor.fetchall()
        return rows

    def close_connection(self):
        """
        Closes the connection to the database.
        """
        self._conn.close()
