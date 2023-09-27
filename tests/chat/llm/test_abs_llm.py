"""
Test the AbsLlm class.
"""
import unittest

from pykoi.chat.llm.abs_llm import AbsLlm


class DummyLlm(AbsLlm):
    """
    Dummy class for testing the abstract base class AbsLlm.
    """

    def predict(self, message: str):
        return f"Q: {message}, A: N/A."


class TestAbsLlm(unittest.TestCase):
    """
    Test the AbsLlm class.
    """

    def test_predict_abstract_method(self):
        """
        Test whether the abstract method `predict` raises NotImplementedError
        """

        test_message = "test"
        llm = DummyLlm()
        self.assertEqual(llm.predict(test_message), f"Q: {test_message}, A: N/A.")

    def test_docstring(self):
        """
        Test whether the class has a docstring
        """
        self.assertIsNotNone(AbsLlm.__doc__)


if __name__ == "__main__":
    unittest.main()
