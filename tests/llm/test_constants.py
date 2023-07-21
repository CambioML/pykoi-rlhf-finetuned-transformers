"""
Test the constants of the LLM module.
"""
import unittest

from plotano.llm.constants import LlmName


class TestLlmName(unittest.TestCase):
    """
    Test the LlmName enum.
    """

    def test_enum_values(self):
        """
        Test whether the enum values are defined correctly
        """
        self.assertEqual(LlmName.OPENAI.value, "openai")
        self.assertEqual(LlmName.HUGGINGFACE.value, "huggingface")

    def test_enum_attributes(self):
        """
        Test whether the enum attributes are defined correctly
        """
        self.assertEqual(LlmName.OPENAI.name, "OPENAI")
        self.assertEqual(LlmName.HUGGINGFACE.name, "HUGGINGFACE")


if __name__ == "__main__":
    unittest.main()
