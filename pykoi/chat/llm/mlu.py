"""MLU HF model."""
from transformers import GenerationConfig
from pykoi.chat.llm.abs_llm import AbsLlm

from transformers import GenerationConfig


class MLUWrapper(AbsLlm):
    model_source = "mlu_trainer"

    def __init__(self, trainer, tokenizer, name=None):
        self._trainer = trainer
        self._model = trainer.model
        self._tokenizer = tokenizer
        self._name = name
        self._model.to("cuda:0")
        self._model.eval()
        super().__init__()

    @property
    def name(self):
        if self._name:
            return self._name
        return "_".join([str(MLUWrapper.model_source), "trainer_model"])

    def predict(self, message: str, num_of_response: int = 1):
        MAX_RESPONSE = 100
        prompt_template = """Below is a sentence that you need to complete. Write a response that appropriately completes the request. Sentence: {instruction}\n Response:"""
        answer_template = """{response}"""

        generation_output = self._model.generate(
            input_ids=self._tokenizer(
                prompt_template.format(instruction=message), return_tensors="pt"
            )["input_ids"].cuda(),
            generation_config=GenerationConfig(
                do_sample=False, num_beams=2
            ),  # Match the standalone function
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=MAX_RESPONSE,
            num_return_sequences=num_of_response,
        )

        response = [
            self._tokenizer.decode(seq, skip_special_tokens=True)
            for seq in generation_output.sequences
        ]
        response = [resp.split("\n")[1] for resp in response if "\n" in resp]
        return response
