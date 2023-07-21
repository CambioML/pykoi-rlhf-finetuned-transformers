import json
import os
import unittest
from dataclasses import asdict

from plotano.rlhf.rlhf import RLHFConfig, read_json_file


class TestRLHFConfig(unittest.TestCase):
    def test_read_json_file_valid(self):
        """
        Test reading a valid json file.
        """
        json_content = {"key": "value"}
        with open("test.json", "w") as file:
            json.dump(json_content, file)

        data = read_json_file("test.json")
        self.assertEqual(data, json_content)

        os.remove("test.json")

    def test_read_json_file_invalid(self):
        """
        Test reading an invalid/nonexistent json file.
        """
        with self.assertRaises(FileNotFoundError):
            read_json_file("nonexistent.json")

    def test_RLHFConfig_default(self):
        """
        Test default initialization of RLHFConfig.
        """
        config = RLHFConfig()
        config_dict = asdict(config)

        # Check a subset of the fields to verify they are correctly initialized
        self.assertEqual(config_dict["base_model_path"], "meta-llama/Llama-2-7b-hf")
        self.assertEqual(config_dict["dataset_type"], "csv")
        self.assertEqual(config_dict["train_test_split_ratio"], 0.1)
        self.assertEqual(config_dict["shuffle_buffer"], 5000)
        # Add checks for the rest of the fields as needed

    def test_RLHFConfig_custom(self):
        """
        Test custom initialization of RLHFConfig.
        """
        custom_values = {
            "base_model_path": "custom_model",
            "dataset_type": "huggingface",
            "train_test_split_ratio": 0.2,
            "shuffle_buffer": 6000
            # Add custom values for the rest of the fields as needed
        }
        config = RLHFConfig(**custom_values)
        config_dict = asdict(config)

        # Check that the fields are correctly initialized with the custom values
        self.assertEqual(config_dict["base_model_path"], "custom_model")
        self.assertEqual(config_dict["dataset_type"], "huggingface")
        self.assertEqual(config_dict["train_test_split_ratio"], 0.2)
        self.assertEqual(config_dict["shuffle_buffer"], 6000)
        # Add checks for the rest of the fields as needed


if __name__ == "__main__":
    unittest.main()
