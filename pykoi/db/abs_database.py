"""abs database"""
import abc
import sqlite3
import threading

from typing import (
    List,
    Tuple
)


class AbsDatabase:
    """Base Database class"""

    def __init__(self,
                 db_file: str,
                 debug: bool = False) -> None:
        """
        Initializes a new instance of the BaseDatabase class.

        Args:
            db_file (str): The path to the SQLite database file.
            debug (bool, optional): Whether to print debug messages. Defaults to False.
        """
        self._db_file = db_file
        self._debug = debug
        self._local = threading.local()  # Thread-local storage
        self._lock = threading.Lock()  # Lock for concurrent write operations

    def get_connection(self) -> sqlite3.Connection:
        """Returns the thread-local database connection"""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(self._db_file)
        return self._local.connection

    def get_cursor(self) -> sqlite3.Cursor:
        """Returns the thread-local database cursor"""
        if not hasattr(self._local, "cursor"):
            self._local.cursor = self.get_connection().cursor()
        return self._local.cursor

    def create_table(self, query: str) -> None:
        """
        Creates the table if it does not already exist in the database.

        Args:
            query (str): The SQL query to create the table.
        """

        with self._lock:
            cursor = self.get_cursor()
            cursor.execute(query)
            self.get_connection().commit()

        if self._debug:
            rows = self.retrieve_all()
            print("Table contents after creating table:")
            self.print_table(rows)

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

    @abc.abstractmethod
    def insert(self, **kwargs) -> None:
        """
        Inserts into the database.

        Args:
            kwargs (dict): The key-value pairs to insert into the database.
        """
        raise NotImplementedError("Insert method must be implemented by subclasses.")

    @abc.abstractmethod
    def update(self, **kwargs) -> None:
        """
        Updates the database.

        Args:
            kwargs (dict): The key-value pairs to update in the database.
        """
        raise NotImplementedError("Update method must be implemented by subclasses.")

    def retrieve_all(self) -> List[Tuple]:
        """
        Retrieves all pairs from the database.
        """
        raise NotImplementedError("Retrieve method must be implemented by subclasses.")

    @abc.abstractmethod
    def print_table(self, rows: str) -> None:
        """
        Prints the table to the console.

        Args:
            rows (str): The rows to print.
        """
        raise NotImplementedError("Print method must be implemented by subclasses.")
