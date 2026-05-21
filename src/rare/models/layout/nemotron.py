"""Nemotron Page Elements v3 backend (YOLOX-based)."""

from __future__ import annotations

from rare.models.registry import register


@register("layout", "nemotron")
class NemotronBackend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        self._model = lp.NemotronLayoutModel(**config) if config else lp.NemotronLayoutModel()

    def detect(self, image):
        return self._model.detect(image)
