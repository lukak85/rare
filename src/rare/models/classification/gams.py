from __future__ import annotations

import torch
from transformers import pipeline, BitsAndBytesConfig

from rare.models.registry import register


@register("classification", "gams")
class GamsClassification:
    """Capitalised-run detector with an optional name list.

    Config keys: ``names`` (iterable of known person names), ``label``
    (label for gazetteer hits, default ``PER``), ``min_tokens``.
    """

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        self.default_label = cfg.get("label", "PER")
        self.min_tokens = int(cfg.get("min_tokens", 2))
        self._keys: dict[str, str] = {}
        self.add_names(cfg.get("names") or [])

        model_id = "cjvt/GaMS3-12B-Instruct"

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

        self.classes = [
            "recenzija", "novica", "kviz", "pisma", "naslov", "reklama", "informativni členek", "intervju"
        ]

    def _prompt_class(self, text: str) -> str:
        prompt = (
            f"Izmed podanih kategorij za sledeče besedilo izberi eno izmed kategorij:"
            f"{', '.join(self.classes)}"
            f"Besedilo: {text}"
        )

        message = [{"role": "user", "content": prompt}]

        response = self.model(message, max_new_tokens=8148)
        return response[0]["generated_text"][-1]["content"]

    def _extract_class(self, names, label: str = "PER") -> str:
        return ""

    def classify(self, text: str) -> str:
        return self._extract_class(self._prompt_class(text))
