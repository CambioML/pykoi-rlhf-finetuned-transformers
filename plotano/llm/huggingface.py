"""Hubbingface model for Language Model (LLM)."""
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer)

from plotano.llm.abs_llm import AbsLlm


class HuggingfaceModel(AbsLlm):
    """
    This class is a wrapper for the Huggingface model for Language Model (LLM) Chain.
    It inherits from the abstract base class AbsLlm.
    """

    def __init__(self,
                 pretrained_model_name_or_path: str,
                 trust_remote_code: bool = True,
                 load_in_8bit: bool = True,
                 max_length: int = 100,
                 device_map: str = "auto"):
        """
        Initialize the Huggingface model.

        Args:
            pretrained_model_name_or_path (str): The name or path of the pretrained model.
            trust_remote_code (bool): Whether to trust the remote code. Default is True.
            load_in_8bit (bool): Whether to load the model in 8-bit. Default is True.
            max_length (int): The maximum length for the model. Default is 100.
            device_map (str): The device map for the model. Default is "auto".
        """
        # running on cpu can be slow!!!
        print("[HuggingfaceModel] loading model...")
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            load_in_8bit=load_in_8bit,
            device_map=device_map)
        print("[HuggingfaceModel] loading tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            load_in_8bit=load_in_8bit,
            device_map=device_map)
        self._max_length = max_length
        super().__init__()

    def predict(self,
                message: str):
        """
        Predict the next word based on the input message.

        Args:
            message (str): The input message for the model.

        Returns:
            str: The predicted next word.
        """
        print("HuggingfaceModel] encode...")
        input_ids = self._tokenizer.encode(message,
                                           return_tensors="pt")
        print("HuggingfaceModel] generate...")
        output_ids = self._model.generate(input_ids,
                                          max_length=self._max_length)
        print("HuggingfaceModel] decode...")
        response = self._tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True)
        return response
