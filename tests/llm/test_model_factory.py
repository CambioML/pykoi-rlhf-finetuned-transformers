import unittest
from unittest.mock import patch
from pykoi.llm.constants import ModelSource
from pykoi.llm.factory import ModelFactory


class TestModelFactory(unittest.TestCase):
    @patch("pykoi.llm.factory.OpenAIModel")
    def test_create_model_openai(self, mock_model):
        """
        Test that the factory creates an OpenAIModel instance when the model
        """
        ModelFactory.create_model(ModelSource.OPENAI, model_name="gpt3")
        mock_model.assert_called_once_with(model_name="gpt3")

    @patch("pykoi.llm.factory.HuggingfaceModel")
    def test_create_model_huggingface(self, mock_model):
        """
        Test that the factory creates a HuggingfaceModel instance when the model
        """
        ModelFactory.create_model(ModelSource.HUGGINGFACE, model_name="gpt2")
        mock_model.assert_called_once_with(model_name="gpt2")

    @patch("pykoi.llm.factory.PeftHuggingfacemodel")
    def test_create_model_peft_huggingface(self, mock_model):
        """
        Test that the factory creates a PeftHuggingfaceModel instance when the model
        """
        ModelFactory.create_model(ModelSource.PEFT_HUGGINGFACE, model_name="gpt2")
        mock_model.assert_called_once_with(model_name="gpt2")

    def test_create_model_invalid_model_source(self):
        """
        Test that the factory raises a ValueError when the model source is invalid.
        """
        with self.assertRaises(ValueError):
            ModelFactory.create_model("INVALID_SOURCE")


if __name__ == "__main__":
    unittest.main()
