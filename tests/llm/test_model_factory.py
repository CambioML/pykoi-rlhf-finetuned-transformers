"""
Test the ModelFactory class.
"""

import unittest
from unittest.mock import MagicMock, patch

from pykoi.llm.model_factory import LlmName, ModelFactory


class TestModelFactory(unittest.TestCase):
    """
    Test the ModelFactory class.
    """

    def test_create_model_openai(self):
        """
        Test creating an OpenAI model instance.
        """
        model_name = LlmName.OPENAI

        # Mock the OpenAIModel behavior
        mock_openai_model = MagicMock()
        openaimodel_mock = MagicMock(return_value=mock_openai_model)

        # Patch the OpenAIModel class to use the mocked version
        with patch("pykoi.llm.model_factory.OpenAIModel", openaimodel_mock):
            result = ModelFactory.create_model(model_name)

        # Check if the OpenAIModel class was called with the correct arguments
        openaimodel_mock.assert_called_once()
        self.assertEqual(result, mock_openai_model)

    def test_create_model_huggingface(self):
        """
        Test creating a Huggingface model instance.
        """
        model_name = LlmName.HUGGINGFACE

        # Mock the HuggingfaceModel behavior
        mock_huggingface_model = MagicMock()
        huggingface_model_mock = MagicMock(return_value=mock_huggingface_model)

        # Patch the HuggingfaceModel class to use the mocked version
        with patch(
            "pykoi.llm.model_factory.HuggingfaceModel", huggingface_model_mock
        ):
            result = ModelFactory.create_model(model_name)

        # Check if the HuggingfaceModel class was called with the correct arguments
        huggingface_model_mock.assert_called_once()
        self.assertEqual(result, mock_huggingface_model)

    def test_create_model_invalid_name(self):
        """
        Test creating a model with an invalid name.
        """
        model_name = "invalid_model_name"

        # Assert that a ValueError is raised when an invalid model name is provided
        with self.assertRaises(ValueError):
            ModelFactory.create_model(model_name)


if __name__ == "__main__":
    unittest.main()
