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

    @patch("pykoi.application.Flask")
    @patch("pykoi.application.CORS")
    @patch("pykoi.application.send_from_directory")
    def test_run(self, send_from_directory_mock, CORS_mock, Flask_mock):
        """
        Tests running the application.
        """
        self.app.data_sources = {
            "data_source1": MagicMock(),
            "data_source2": MagicMock(),
        }
        self.app.components = [
            {"id": "test_id", "component": MagicMock(), "svelte_component": "Chatbot"}
        ]

        self.app.run()

        # Check if Flask and CORS were called correctly
        Flask_mock.assert_called_once()
        CORS_mock.assert_called_once_with(Flask_mock.return_value)

        # Check if routes are created correctly
        self.assertEqual(len(Flask_mock.return_value.route.call_args_list), 11)

        # Check if data routes are created correctly
        Flask_mock.return_value.route.assert_any_call(
            "/data/data_source1", methods=["GET"], endpoint="data_source1"
        )
        Flask_mock.return_value.route.assert_any_call(
            "/data/data_source2", methods=["GET"], endpoint="data_source2"
        )

        # Check if chatbot routes are created correctly
        Flask_mock.return_value.route.assert_any_call(
            "/chat/<message>", methods=["POST"]
        )
        Flask_mock.return_value.route.assert_any_call(
            "/chat/qa_table/update", methods=["POST"]
        )
        Flask_mock.return_value.route.assert_any_call(
            "/chat/qa_table/close", methods=["GET"]
        )

        # Check if the base and home routes are created correctly
        if os.path.exists("frontend/dist"):
            Flask_mock.return_value.route.assert_any_call("/", methods=["GET"])
            Flask_mock.return_value.route.assert_any_call(
                "/<path:path>", methods=["GET"]
            )

        # Check if the app.run() method is called with the correct arguments
        app_run_args = Flask_mock.return_value.run.call_args[1]
        self.assertEqual(app_run_args["debug"], False)


if __name__ == "__main__":
    unittest.main()
