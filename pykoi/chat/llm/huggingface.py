"""Huggingface model for Language Model (LLM)."""
from transformers import AutoModelForCausalLM, AutoTokenizer

from pykoi.chat.llm.abs_llm import AbsLlm


class HuggingfaceModel(AbsLlm):
    """
    This class is a wrapper for the Huggingface model for Language Model (LLM) Chain.
    It inherits from the abstract base class AbsLlm.
    """

    model_source = "huggingface"

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        name: str = None,
        trust_remote_code: bool = True,
        load_in_8bit: bool = True,
        max_length: int = 100,
        device_map: str = "auto",
    ):
        """
        Initialize the Huggingface model.

        Args:
            pretrained_model_name_or_path (str): The name or path of the pretrained model.
            name (str): The name of the model
            trust_remote_code (bool): Whether to trust the remote code. Default is True.
            load_in_8bit (bool): Whether to load the model in 8-bit. Default is True.
            max_length (int): The maximum length for the model. Default is 100.
            device_map (str): The device map for the model. Default is "auto".
        """
        if pretrained_model_name_or_path != "":
            # running on cpu can be slow!!!
            print("[HuggingfaceModel] loading model...")
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                load_in_8bit=load_in_8bit,
                device_map=device_map,
            )
            print("[HuggingfaceModel] loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                load_in_8bit=load_in_8bit,
                device_map=device_map,
            )
        self._max_length = max_length
        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        self._name = name
        super().__init__()

    @property
    def name(self):
        if self._name:
            return self._name
        return "_".join(
            [
                str(HuggingfaceModel.model_source),
                str(self._pretrained_model_name_or_path),
                str(self._max_length),
            ]
        )

    def predict(self, message: str, num_of_response: int = 1):
        """
        Predict the next word based on the input message.

        Args:
            message (str): The input message for the model.
            num_of_response (int): The number of response to generate. Default is 1.

        Returns:
            List[str]: List of response.
        """
        # TODO: need to refractor and include all the derivatives of dolly family
        if "dolly" in self._pretrained_model_name_or_path:
            from pykoi.chat.llm.instruct_pipeline import (
                InstructionTextGenerationPipeline,
            )

            generate_text = InstructionTextGenerationPipeline(
                model=self._model, tokenizer=self._tokenizer
            )
            res = generate_text(message)
            response = [res[0]["generated_text"]]
        # all other models except dolly family
        else:
            print("[HuggingfaceModel] encode...")
            input_ids = self._tokenizer.encode(message, return_tensors="pt")
            input_ids = input_ids.to("cuda")
            print("[HuggingfaceModel] generate...")
            output_ids = self._model.generate(
                input_ids,
                max_length=self._max_length,
                num_return_sequences=num_of_response,
                do_sample=True,
                temperature=0.3,
            )
            print("[HuggingfaceModel] decode...")
            response = [
                self._tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]

            response = [
                "\n".join(resp.split("\n")[1:]) for resp in response if "\n" in resp
            ]

        return response

    @classmethod
    def create(cls, model, tokenizer, name=None, max_length=100):
        """
        Initialize the Huggingface model with given model and tokenizer.

        Args:
            model: Pre-loaded model instance from Huggingface.
            tokenizer: Pre-loaded tokenizer instance from Huggingface.
            name (str): The name of the model
            max_length (int): The maximum length for the model. Default is 100.

        Returns:
            An instance of HuggingfaceModel.
        """
        instance = cls(
            pretrained_model_name_or_path="",  # This won't be used since we are directly assigning the model & tokenizer
            name=name,
            trust_remote_code=True,  # Setting defaults, but these won't be used
            load_in_8bit=True,
            max_length=max_length,
            device_map="auto",
        )
        instance._model = model
        instance._tokenizer = tokenizer
        return instance
