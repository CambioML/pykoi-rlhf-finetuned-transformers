"""Tests for the base module."""
import unittest

from pykoi.component.base import Component, DataSource


class TestDataSource(unittest.TestCase):
    """
    Unit test class for the DataSource class.

    This class tests the initialization of the DataSource class by creating a DataSource instance
    with a specific id and a fetch function. It then asserts that the id and the data fetched by
    the fetch function are as expected.

    Attributes:
        fetch_func (function): A function that returns the data to be fetched.
        ds (DataSource): An instance of the DataSource class.

    Methods:
        test_init: Tests the initialization of the DataSource class.
    """

    def test_init(self):
        """
        Tests the initialization of the DataSource class.

        This method creates a DataSource instance with a specific id and a fetch function.
        It then asserts that the id and the data fetched by the fetch function are as expected.
        """

        def fetch_func():
            return "data"

        data_source = DataSource("test_id", fetch_func)

        self.assertEqual(data_source.id, "test_id")
        self.assertEqual(data_source.fetch_func(), "data")


class TestComponent(unittest.TestCase):
    """
    Unit test class for the Component class.

    This class tests the initialization of the Component class by creating a Component instance
    with a specific fetch function, a svelte component, and properties. It then asserts that the id,
    the data fetched by the fetch function, the svelte component, and the properties are as expected.

    Attributes:
        fetch_func (function): A function that returns the data to be fetched.
        comp (Component): An instance of the Component class.

    Methods:
        test_init: Tests the initialization of the Component class.
    """

    def test_init(self):
        """
        Tests the initialization of the Component class.

        This method creates a Component instance with a specific fetch function, a svelte component,
        and properties. It then asserts that the id, the data fetched by the fetch function, the svelte
        component, and the properties are as expected.
        """

        def fetch_func():
            return "data"

        comp = Component(fetch_func, "TestComponent", prop1="value1", prop2="value2")

        self.assertIsNotNone(comp.id)
        self.assertEqual(comp.data_source.fetch_func(), "data")
        self.assertEqual(comp.svelte_component, "TestComponent")
        self.assertEqual(comp.props, {"prop1": "value1", "prop2": "value2"})


if __name__ == "__main__":
    unittest.main()
