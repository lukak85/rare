from __future__ import annotations

from rare.models.registry import register


PRED_CATEGORY_MAPS: dict[str, dict[str, str]] = {
    "DocLayNet": {
        "Caption":        "figure_caption",
        "Footnote":       "page_footnote",
        "Formula":        "equation_isolated",
        "List-item":      "text_block",
        "Page-footer":    "footer",
        "Page-header":    "header",
        "Picture":        "figure",
        "Section-header": "title",
        "Table":          "table",
        "Text":           "text_block",
        "Title":          "title",
    },
}

@register("layout", "detr")
class RFDETRBackend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        self._model = lp.DETRLayoutModel(**config) if config else lp.DETRLayoutModel()
        if config is not None:
            label_map = config.get("label_map")
        else:
            label_map = "DocLayNet"
        self.pred_category_map = PRED_CATEGORY_MAPS.get(label_map)
        self.label_map = self._model.label_map

    def detect(self, image):
        return self._model.detect(image)
