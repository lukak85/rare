from __future__ import annotations

import torch
from transformers import pipeline, BitsAndBytesConfig

from rare.models.registry import register


@register("classification", "gams")
class GamsClassification:
    """Editorial-genre classifier prompting GaMS-3 12B Instruct in Slovenian.

    Config keys: ``model`` (checkpoint, default ``cjvt/GaMS3-12B-Instruct``),
    ``classes`` (the genre list to choose from), ``max_new_tokens``.
    """

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        self.max_new_tokens = int(cfg.get("max_new_tokens", 32))

        model_id = cfg.get("model", "cjvt/GaMS3-12B-Instruct")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = pipeline(
            "text-generation",
            model=model_id,
            # device_map="auto",
            model_kwargs={"quantization_config": quant_config},
            device_map={"": 0},  # force everything onto GPU 0; errors loudly instead of offloading
            torch_dtype=torch.bfloat16,

        )

        self.classes = list(cfg.get("classes") or [
            "recenzija", "novica", "kviz", "pisma", "naslov", "reklama", "informativni členek", "intervju"
        ])

    def _prompt_class(self, text: str) -> str:
        prompt = (
            f"Izmed podanih kategorij za sledeče besedilo izberi eno izmed kategorij: "
            f"{', '.join(self.classes)}. Odgovori samo z imenom kategorije.\n"
            f"Besedilo: {text}"
        )

        message = [{"role": "user", "content": prompt}]

        response = self.model(message, max_new_tokens=self.max_new_tokens)
        return response[0]["generated_text"][-1]["content"]

    def classify(self, text: str) -> str:
        """Return the model's reply verbatim.

        Mapping free-form Slovenian back onto `self.classes` is the caller's
        job (`rare.link.classify`), which reads `classes` off this object — so
        the same matching serves any generative backend registered here.
        """
        return self._prompt_class(text)
