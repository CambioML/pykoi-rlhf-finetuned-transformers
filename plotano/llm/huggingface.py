import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer)

from plotano.llm.abs_llm import AbsLlm


class HuggingfaceModel(AbsLlm):
    """Huggingface model wrapper for LLMChain."""

    def __init__(self,
                 pretrained_model_name_or_path: str,
                 trust_remote_code: bool = True,
                 load_in_8bit: bool = True,
                 max_length: int = 100,
                 device_map: str = "auto"):
        """Initialize the Huggingface model."""
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
        """Predict the next word."""
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
