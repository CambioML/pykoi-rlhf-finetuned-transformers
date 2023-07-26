"""Huggingface PEFT model for Language Model (LLM)."""
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from pykoi.llm.abs_llm import AbsLlm


class PeftHuggingfacemodel(AbsLlm):
    """
    This class is a wrapper for the Huggingface PEFT model for Language Model (LLM).

    Attributes:
        _model (PeftModel): The PEFT model.
        _tokenizer (AutoTokenizer): The tokenizer for the model.
        _max_length (int): The maximum length of the generated text.
    """
    model_source = "peft_huggingface"

    def __init__(
        self,
        base_model_path: str,
        lora_model_path: str,
        trust_remote_code: bool = True,
        load_in_8bit: bool = True,
        max_length: int = 100,
        device_map: str = "auto",
    ):
        """
        The constructor for PeftHuggingfacemodel class.

        Args:
            base_model_path (str): The path to the base model.
            lora_model_path (str): The path to the lora model.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to True.
            load_in_8bit (bool, optional): Whether to load in 8bit. Defaults to True.
            max_length (int, optional): The maximum length of the generated text. Defaults to 100.
            device_map (str, optional): The device map. Defaults to "auto".
        """
        print("[HuggingfaceModel] loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            return_dict=True,
            torch_dtype=torch.float16,
            trust_remote_code=trust_remote_code,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
        )

        print("[HuggingfaceModel] loading perf model...")
        self._model = PeftModel.from_pretrained(
            model=base_model,
            model_id=lora_model_path)

        print("[HuggingfaceModel] loading tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=base_model_path,
            trust_remote_code=trust_remote_code,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
        )
        self._max_length = max_length
        self._base_model_path = base_model_path
        self._lora_model_path = lora_model_path
        self._model.to("cuda")
        self._model.eval()
        super().__init__()

    @property
    def name(self):
        return "_".join([
            str(PeftHuggingfacemodel.model_source),
            str(self._base_model_path),
            str(self._lora_model_path),
            str(self._max_length)
        ])

    def predict(self, message: str, num_of_response: int = 1):
        """
        Predict the next word based on the input message.

        Args:
            message (str): The input message for the model.
            num_of_response (int, optional): The number of response to generate. Default is 1.

        Returns:
            List[str]: List of response.
        """
        print("[HuggingfaceModel] encode...")
        input_ids = self._tokenizer.encode(message, return_tensors="pt")
        input_ids = input_ids.to("cuda")
        print("[HuggingfaceModel] generate...")
        output_ids = self._model.generate(
            input_ids=input_ids,
            max_length=self._max_length,
            num_return_sequences=num_of_response,
            do_sample=True,
            temperature=0.3,
        )
        print("[HuggingfaceModel] decode...")
        response = [
            self._tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
        ]
        print("response: ", response)

        response = [resp.split("\n")[1] for resp in response if "\n" in resp]

        return response
