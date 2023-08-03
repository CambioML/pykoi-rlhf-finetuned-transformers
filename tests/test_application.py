"""
Test the Application class.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from pykoi.application import Application


class TestApplication(unittest.TestCase):
    """
    Unit test class for the Application class.
    """

    def setUp(self):
        self.app = Application(share=False, debug=False)

    def test_add_component(self):
        """
        Tests adding a component to the application.
        """
        component = MagicMock()
        component.data_source = MagicMock()
        component.id = "test_component"

        self.app.add_component(component)

        # Check if the component and its data source are added to the application
        self.assertIn("test_component", self.app.data_sources)
        self.assertIn(component, [c["component"] for c in self.app.components])


if __name__ == "__main__":
    unittest.main()
