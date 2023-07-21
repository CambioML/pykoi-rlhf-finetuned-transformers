"""
Test the OpenAIModel class
"""
import unittest
from unittest.mock import MagicMock, patch

from plotano.llm.openai import OpenAIModel


class TestOpenAIModel(unittest.TestCase):
    """
    Test the OpenAIModel class
    """

    def test_predict(self):
        """
        Test the predict method of the OpenAIModel class
        """
        # Test predicting the next word based on a given message
        message = "What is the meaning of life?"
        predicted_word = "42"

        # Mock the OpenAI.Completion.create behavior
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].text = f"Answer: {predicted_word}"
        openai_completion_create_mock = MagicMock(return_value=mock_response)

        # Patch the OpenAI.Completion.create method to use the mocked version
        with patch(
            "plotano.llm.openai.openai.Completion.create", openai_completion_create_mock
        ):
            openai_model = OpenAIModel(
                api_key="fake_api_key",
                engine="davinci",
                max_tokens=100,
                temperature=0.5,
            )
            result = openai_model.predict(message, 1)

        # Check if the OpenAI.Completion.create method was called with the correct arguments
        openai_completion_create_mock.assert_called_once_with(
            engine="davinci",
            prompt=f"Question: {message}\nAnswer:",
            max_tokens=100,
            n=1,
            stop="\n",
            temperature=0.5,
        )
        self.assertEqual(result[0], f"Answer: {predicted_word}")


if __name__ == "__main__":
    unittest.main()
