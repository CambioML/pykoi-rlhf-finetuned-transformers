"""
Test the RankingDatabase class
"""
import os
import sqlite3
import unittest

from plotano.db.ranking_database import RankingDatabase

# Define a temporary database file for testing
TEST_DB_FILE = "test_ranking.db"


class TestRankingDatabase(unittest.TestCase):
    """
    Test the RankingDatabase class
    """

    def setUp(self):
        # Create a temporary database for testing
        self.ranking_db = RankingDatabase(db_file=TEST_DB_FILE, debug=True)

    def tearDown(self):
        # Remove the temporary database and close connections after each test
        self.ranking_db.close_connection()
        os.remove(TEST_DB_FILE)

    def test_create_table(self):
        """
        Test whether the table is created correctly.
        """
        conn = sqlite3.connect(TEST_DB_FILE)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ranking'"
        )
        table_exists = cursor.fetchone()

        self.assertTrue(table_exists)

        # Clean up
        cursor.close()
        conn.close()

    def test_insert_and_retrieve_ranking(self):
        """
        Test inserting and retrieving a ranking entry
        """
        question = "Which fruit is your favorite?"
        up_ranking_answer = "Apple"
        low_ranking_answer = "Banana"

        # Insert data and get the ID
        ranking_id = self.ranking_db.insert_ranking(
            question, up_ranking_answer, low_ranking_answer
        )

        # Retrieve the data
        rows = self.ranking_db.retrieve_all_question_answers()

        # Check if the data was inserted correctly
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], ranking_id)
        self.assertEqual(rows[0][1], question)
        self.assertEqual(rows[0][2], up_ranking_answer)
        self.assertEqual(rows[0][3], low_ranking_answer)

    def test_save_to_csv(self):
        """
        Test saving data to a CSV file
        """
        question1 = "Which fruit is your favorite?"
        up_ranking_answer1 = "Apple"
        low_ranking_answer1 = "Banana"
        question2 = "Which country would you like to visit?"
        up_ranking_answer2 = "Japan"
        low_ranking_answer2 = "Italy"

        # Insert data
        self.ranking_db.insert_ranking(
            question1, up_ranking_answer1, low_ranking_answer1
        )
        self.ranking_db.insert_ranking(
            question2, up_ranking_answer2, low_ranking_answer2
        )

        # Save to CSV
        self.ranking_db.save_to_csv("test_csv_file.csv")

        # Check if the CSV file was created and contains the correct data
        self.assertTrue(os.path.exists("test_csv_file.csv"))

        with open("test_csv_file.csv", "r") as file:
            lines = file.readlines()

        # Verify the CSV file content
        self.assertEqual(len(lines), 3)  # Header + 2 rows
        self.assertEqual(
            lines[0].strip(), "ID,Question,Up Ranking Answer,Low Ranking Answer"
        )
        self.assertEqual(
            lines[1].strip(),
            f"1,{question1},{up_ranking_answer1},{low_ranking_answer1}",
        )
        self.assertEqual(
            lines[2].strip(),
            f"2,{question2},{up_ranking_answer2},{low_ranking_answer2}",
        )

        # Clean up
        os.remove("test_csv_file.csv")


if __name__ == "__main__":
    unittest.main()
