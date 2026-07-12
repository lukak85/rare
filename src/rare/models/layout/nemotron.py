"""Nemotron Page Elements v3 backend (YOLOX-based)."""

from __future__ import annotations

from rare.models.registry import register


@register("layout", "nemotron")
class NemotronBackend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        self._model = lp.NemotronLayoutModel(**config) if config else lp.NemotronLayoutModel()

        self.pred_category_map = {
            "table": "table",
            "chart": "figure",
            "title": "title",
            "infographic": "figure",
            "text": "text_block",
            "header_footer": "header",
        }

        self.label_map = {
            0: "table",
            1: "chart",
            2: "title",
            3: "infographic",
            4: "text",
            5: "header_footer",
        }

    def detect(self, image):
        return self._model.detect(image)
