"""
Test the constants of the LLM module.
"""
import unittest

from pykoi.chat.llm.constants import ModelSource


class TestLlmName(unittest.TestCase):
    """
    Test the ModelSource enum.
    """

    def test_enum_values(self):
        """
        Test whether the enum values are defined correctly
        """
        self.assertEqual(ModelSource.OPENAI.value, "openai")
        self.assertEqual(ModelSource.HUGGINGFACE.value, "huggingface")

    def test_enum_attributes(self):
        """
        Test whether the enum attributes are defined correctly
        """
        self.assertEqual(ModelSource.OPENAI.name, "OPENAI")
        self.assertEqual(ModelSource.HUGGINGFACE.name, "HUGGINGFACE")


if __name__ == "__main__":
    unittest.main()
