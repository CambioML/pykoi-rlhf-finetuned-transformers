import unittest
from unittest.mock import patch, Mock
from transformers import AutoModelForCausalLM, AutoTokenizer

from pykoi.llm.abs_llm import AbsLlm
from pykoi.llm.huggingface import HuggingfaceModel


class TestHuggingfaceModel(unittest.TestCase):
    @patch.object(AutoModelForCausalLM, "from_pretrained")
    @patch.object(AutoTokenizer, "from_pretrained")
    def setUp(self, mock_model, mock_tokenizer):
        self.model_name = "gpt2"
        self.mock_model = mock_model
        self.mock_tokenizer = mock_tokenizer

        # Mocking the pretrained model and tokenizer
        self.mock_model.return_value = Mock()
        self.mock_tokenizer.return_value = Mock()

        self.huggingface_model = HuggingfaceModel(
            pretrained_model_name_or_path=self.model_name
        )

    def test_name(self):
        expected_name = f"{HuggingfaceModel.model_source}_{self.model_name}_100"
        self.assertEqual(self.huggingface_model.name, expected_name)

    @patch.object(HuggingfaceModel, "predict")
    def test_predict(self, mock_predict):
        mock_predict.return_value = ["Hello, how can I assist you today?"]
        message = "Hello, chatbot!"
        num_of_response = 1
        response = self.huggingface_model.predict(message, num_of_response)
        mock_predict.assert_called_once_with(message, num_of_response)
        self.assertEqual(response, ["Hello, how can I assist you today?"])


if __name__ == "__main__":
    unittest.main()
