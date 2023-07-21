"""Test the QuestionAnswerDatabase class"""
import datetime
import os
import sqlite3
import unittest

from plotano.db.qa_database import QuestionAnswerDatabase

# Define a temporary database file for testing
TEST_DB_FILE = "test_qd.db"


class TestQuestionAnswerDatabase(unittest.TestCase):
    """
    Test the QuestionAnswerDatabase class.
    """

    def setUp(self):
        # Create a temporary database for testing
        self.qadb = QuestionAnswerDatabase(db_file=TEST_DB_FILE, debug=True)

    def tearDown(self):
        # Remove the temporary database and close connections after each test
        self.qadb.close_connection()
        os.remove(TEST_DB_FILE)

    def test_create_table(self):
        """
        Test whether the table is created correctly.
        """
        # Test whether the table is created correctly
        conn = sqlite3.connect(TEST_DB_FILE)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='question_answer'"
        )
        table_exists = cursor.fetchone()

        self.assertTrue(table_exists)

        # Clean up
        cursor.close()
        conn.close()

    def test_insert_and_retrieve_question_answer(self):
        """
        Test inserting and retrieving a question-answer pair
        """
        question = "What is the meaning of life?"
        answer = "42"

        # Insert data and get the ID
        qa_id = self.qadb.insert_question_answer(question, answer)

        # Retrieve the data
        rows = self.qadb.retrieve_all_question_answers()

        # Check if the data was inserted correctly
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], qa_id)
        self.assertEqual(rows[0][1], question)
        self.assertEqual(rows[0][2], answer)
        self.assertEqual(rows[0][3], "n/a")  # Default vote status

    def test_update_vote_status(self):
        """
        Test updating the vote status of a question-answer pair.
        """
        question = "What is the meaning of life?"
        answer = "42"

        # Insert data and get the ID
        qa_id = self.qadb.insert_question_answer(question, answer)

        # Update the vote status
        new_vote_status = "up"
        self.qadb.update_vote_status(qa_id, new_vote_status)

        # Retrieve the data
        rows = self.qadb.retrieve_all_question_answers()

        # Check if the vote status was updated correctly
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], qa_id)
        self.assertEqual(rows[0][3], new_vote_status)

    def test_save_to_csv(self):
        """
        Test saving data to a CSV file
        """
        question1 = "What is the meaning of life?"
        answer1 = "42"
        question2 = "What is the best programming language?"
        answer2 = "Python"

        # Insert data
        timestamp = datetime.datetime.now()
        self.qadb.insert_question_answer(question1, answer1)
        self.qadb.insert_question_answer(question2, answer2)

        # Save to CSV
        self.qadb.save_to_csv("test_csv_file.csv")

        # Check if the CSV file was created and contains the correct data
        self.assertTrue(os.path.exists("test_csv_file.csv"))

        with open("test_csv_file.csv", "r") as file:
            lines = file.readlines()

        # Verify the CSV file content
        timestamp_trim = 10  # Trim 10 characters from the timestamp
        self.assertEqual(len(lines), 3)  # Header + 2 rows
        self.assertEqual(lines[0].strip(), "ID,Question,Answer,Vote Status,Timestamp")
        self.assertEqual(
            lines[1].strip()[:-timestamp_trim],
            f"1,{question1},{answer1},n/a,{timestamp}"[:-timestamp_trim],
        )  # Default vote status
        self.assertEqual(
            lines[2].strip()[:-timestamp_trim],
            f"2,{question2},{answer2},n/a,{timestamp}"[:-timestamp_trim],
        )  # Default vote status

        # Clean up
        os.remove("test_csv_file.csv")


if __name__ == "__main__":
    unittest.main()
