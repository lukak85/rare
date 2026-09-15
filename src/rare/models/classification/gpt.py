from __future__ import annotations

import asyncio

import aiohttp

from rare.models.registry import register


@register("classification", "gpt")
class GPTClassification:

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})

        api_key = cfg.get("api_key")
        base_url = cfg.get("base_url")
        if not api_key:
            raise ValueError("API key must be provided for GPTClassification.")
        if not base_url:
            raise ValueError("Base URL must be provided for GPTClassification.")

        self.base_url = base_url.rstrip("/")
        self.model_name = cfg.get("model", "gpt-5.5")
        self.timeout = float(cfg.get("timeout", 120))

        self.headers = {
            "Authorization": f"Bearer {api_key}",
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
            f"{text}"
        )

        async with session.post(
                url=f"{self.base_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                },
                headers=self.headers,
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"{self.model_name} returned HTTP {response.status}: {body[:500]}")
            result = await response.json()
            return result["choices"][0]["message"]["content"]

    def classify(self, text: str) -> str:
        """Return the model's reply verbatim.

        Mapping free-form Slovenian back onto `self.classes` is the caller's
        job (`rare.link.classify`), which reads `classes` off this object — so
        the same matching serves any generative backend registered here.
        """
        async def main():
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                return await self._prompt_class(session, text)

        return asyncio.run(main())
