from __future__ import annotations

import logging

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

        try:
            self.model = pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={
                    "device_map": "auto",
                    "torch_dtype": torch.bfloat16,
    #                "max_memory": {0: "22GiB", 1: "22GiB"},
                    "attn_implementation": "eager",
                },
            )
        except Exception as e:
            logging.log(e)
            return

        self.classes = list(cfg.get("classes") or [
            "reklama",
            "članek", # "(informativni) članek",
            "naslovnica",
            "dogodki",
            "slike",
            "intervju",
            "pisma",  # "pisma (bralcev)",
            "novice",
            "kviz",
            "recenzija", # records,
            # "posebno",
            "kazalo"
        ])

    def _prompt_class(self, text: str) -> str:
        prompt = (
            f"Izmed podanih kategorij za sledeče besedilo izberi eno izmed kategorij: "
            f"{', '.join(self.classes)}. Odgovori samo z imenom kategorije.\n"
            f"{text}"
        )

        message = [{"role": "user", "content": prompt}]

        #tok = self.model.tokenizer
        #ids = tok.apply_chat_template(message, add_generation_prompt=True)
        #print(self.model.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False))
        #exit()

        response = self.model(message, max_new_tokens=self.max_new_tokens, truncation=True)
        res = response[0]["generated_text"][-1]["content"]
        # print(res)
        return res

    def classify(self, text: str) -> str:
        """Return the model's reply verbatim.

        Mapping free-form Slovenian back onto `self.classes` is the caller's
        job (`rare.link.classify`), which reads `classes` off this object — so
        the same matching serves any generative backend registered here.
        """
        return self._prompt_class(text)
