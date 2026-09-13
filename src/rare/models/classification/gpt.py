from __future__ import annotations

import asyncio

import aiohttp

from rare.models.registry import register


@register("classification", "gams")
class GamsClassification:
    """Editorial-genre classifier prompting GaMS-3 12B Instruct in Slovenian.

    Config keys: ``model`` (checkpoint, default ``cjvt/GaMS3-12B-Instruct``),
    ``classes`` (the genre list to choose from), ``max_new_tokens``.
    """

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})

        api_key = config.get("api_key", None)
        base_url = config.get("base_url", None)
        model_name = config.get("model", "gpt-5.5")

        if api_key is None:
            raise ValueError("API key must be provided for GPTBackend.")
        if base_url is None:
            raise ValueError("Base URL must be provided for GPTBackend.")

        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

        self.max_retries = 0  # set retry times
        self.request_sleep = 5  # set sleep time(seconds)

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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

    async def _prompt_class(self, session: aiohttp.ClientSession, text: str) -> str:
        prompt = (
            f"Izmed podanih kategorij za sledeče besedilo izberi eno izmed kategorij: "
            f"{', '.join(self.classes)}. Odgovori samo z imenom kategorije.\n"
            f"Besedilo: {text}"
        )

        async with session.post(
                url=f"{self.base_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                },
                headers=self.headers
        ) as response:
            await asyncio.sleep(self.request_sleep)
            print(response)

            if response.status == 200:
                result = await response.json()
                print(result)
                return result['choices'][0]['message']['content']

        return None

    def run_classification(self, text: str) -> str:
        """Run the classification synchronously."""
        import asyncio

        async def main():
            async with aiohttp.ClientSession() as session:
                return await self._prompt_class(session, text)

        return asyncio.run(main())

    def classify(self, text: str) -> str:
        """Return the model's reply verbatim.

        Mapping free-form Slovenian back onto `self.classes` is the caller's
        job (`rare.link.classify`), which reads `classes` off this object — so
        the same matching serves any generative backend registered here.
        """
        return self.run_classification(text)
