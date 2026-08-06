from __future__ import annotations

import logging

from rare.doc.schema import Entity
from rare.models.ner.normalize import entity_key
from rare.models.registry import register

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "martinkorelic/rudar-mmbert-slv-ner"

# The model card warns that "simple" fragments subword spans; "average" pools
# them, which is what keeps multi-token surnames intact.
DEFAULT_AGGREGATION = "average"


@register("ner", "rudar-slv")
class RudarSlvNER:
    """Wraps `transformers.pipeline("token-classification")`.

    Config keys: ``model``, ``device``, ``aggregation_strategy``,
    ``batch_size``, ``min_score``, ``max_chars``.
    """

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        self.model_id = cfg.get("model") or DEFAULT_MODEL
        self.device = cfg.get("device")
        self.aggregation_strategy = cfg.get(
            "aggregation_strategy", DEFAULT_AGGREGATION
        )
        self.batch_size = int(cfg.get("batch_size", 16))
        self.min_score = float(cfg.get("min_score", 0.5))
        # Guard against a runaway region blowing past the model's context.
        self.max_chars = int(cfg.get("max_chars", 4000))
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline  # heavy: import only when used

            kwargs = {
                "task": "token-classification",
                "model": self.model_id,
                "aggregation_strategy": self.aggregation_strategy,
            }
            if self.device is not None:
                kwargs["device"] = self.device
            logger.info("loading NER model %s", self.model_id)
            self._pipeline = pipeline(**kwargs)
        return self._pipeline

    def extract(self, texts: list[str]) -> list[list[Entity]]:
        results: list[list[Entity]] = [[] for _ in texts]

        # Only send non-empty texts to the model, but keep the mapping back to
        # the caller's indices so the returned list stays aligned.
        indices = [i for i, t in enumerate(texts) if t and t.strip()]
        if not indices:
            return results

        payload = [texts[i][: self.max_chars] for i in indices]
        ner = self._get_pipeline()
        raw_batches = ner(payload, batch_size=self.batch_size)

        # A single-input call returns one flat list rather than a list of lists.
        if payload and raw_batches and not isinstance(raw_batches[0], list):
            raw_batches = [raw_batches]

        for index, spans in zip(indices, raw_batches):
            results[index] = self._to_entities(spans or [])
        return results

    def _to_entities(self, spans) -> list[Entity]:
        entities = []
        for span in spans:
            score = float(span.get("score", 1.0))
            if score < self.min_score:
                continue
            surface = (span.get("word") or "").strip()
            if not surface:
                continue
            label = str(span.get("entity_group") or span.get("entity") or "")
            # Strip any BIO prefix left by non-aggregating strategies.
            label = label.split("-")[-1].upper()
            entities.append(
                Entity(
                    text=surface,
                    label=label,
                    start=int(span.get("start") or 0),
                    end=int(span.get("end") or 0),
                    score=score,
                    key=entity_key(surface, label),
                )
            )
        return entities
