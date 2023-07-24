"""
Tests for the State class.
"""
import unittest

from pykoi.state import State, Store


class TestState(unittest.TestCase):
    """
    Unit test class for the State class.
    """

    def test_get_attribute(self):
        """
        Tests getting an attribute from the state.
        """
        state = State()
        state.state = {"test_attr": 42}

        self.assertEqual(state.test_attr, 42)

    def test_get_non_existing_attribute(self):
        """
        Tests getting a non-existing attribute from the state.
        """
        state = State()

        with self.assertRaises(AttributeError):
            attr = state.non_existing_attr

    def test_set_attribute(self):
        """
        Tests setting an attribute in the state.
        """
        state = State()
        state.test_attr = 42

        self.assertEqual(state.state["test_attr"], 42)

    def test_call_method(self):
        """
        Tests calling a method from the state.
        """
        state = State()
        state.state = {"test_method": lambda x: x * 2}

        result = state.test_method(3)
        self.assertEqual(result, 6)


class TestStore(unittest.TestCase):
    """
    Unit test class for the Store class.
    """

    def test_increment(self):
        """
        Tests incrementing the count.
        """
        store = Store()
        store.increment()

        self.assertEqual(store.count, 5)

    def test_decrement(self):
        """
        Tests decrementing the count.
        """
        store = Store()
        store.decrement()

        self.assertEqual(store.count, 3)

    def test_hello(self):
        """
        Tests the hello method.
        """
        store = Store()
        hello_msg = store.hello()

        self.assertEqual(hello_msg, "hello jared")


if __name__ == "__main__":
    unittest.main()
