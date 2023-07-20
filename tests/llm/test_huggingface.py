"""
Test the HuggingfaceModel class.
"""
import unittest
from unittest.mock import patch, MagicMock
from plotano.llm.huggingface import HuggingfaceModel


class TestHuggingfaceModel(unittest.TestCase):
    """
    Test the HuggingfaceModel class.
    """

    @patch("plotano.llm.huggingface.AutoModelForCausalLM", autospec=True)
    @patch("plotano.llm.huggingface.AutoTokenizer", autospec=True)
    def test_init(self, mock_tokenizer, mock_model):
        """
        Test the initialization of the HuggingfaceModel class.
        """
        pretrained_model_name_or_path = "gpt2"

        # Mock the AutoModelForCausalLM and AutoTokenizer objects
        model_instance_mock = MagicMock()
        tokenizer_instance_mock = MagicMock()
        mock_model.from_pretrained.return_value = model_instance_mock
        mock_tokenizer.from_pretrained.return_value = tokenizer_instance_mock

        huggingface_model = HuggingfaceModel(pretrained_model_name_or_path)

        # Check if the model and tokenizer were initialized correctly
        mock_model.from_pretrained.assert_called_once_with(
            pretrained_model_name_or_path, trust_remote_code=True, load_in_8bit=True, device_map="auto"
        )
        mock_tokenizer.from_pretrained.assert_called_once_with(
            pretrained_model_name_or_path, trust_remote_code=True, load_in_8bit=True, device_map="auto"
        )
        self.assertEqual(huggingface_model._model, model_instance_mock)
        self.assertEqual(huggingface_model._tokenizer, tokenizer_instance_mock)
        self.assertEqual(huggingface_model._max_length, 100)

    def test_predict(self):
        """
        Test the predict method of the HuggingfaceModel class.
        """
        pretrained_model_name_or_path = "gpt2"
        huggingface_model = HuggingfaceModel(pretrained_model_name_or_path)

        input_message = "This is a test message."
        output_message = "This is the predicted message."

        # Mock the tokenizer and model behavior
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        huggingface_model._tokenizer = mock_tokenizer
        huggingface_model._model = mock_model

        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_model.generate.return_value = [[6, 7, 8, 9, 10]]
        mock_tokenizer.decode.return_value = output_message

        result = huggingface_model.predict(input_message)

        # Check if the tokenizer and model methods were called with the correct arguments
        mock_tokenizer.encode.assert_called_once_with(input_message, return_tensors="pt")
        mock_model.generate.assert_called_once_with([1, 2, 3, 4, 5], max_length=100)
        mock_tokenizer.decode.assert_called_once_with([6, 7, 8, 9, 10], skip_special_tokens=True)

        self.assertEqual(result, output_message)


if __name__ == "__main__":
    unittest.main()
