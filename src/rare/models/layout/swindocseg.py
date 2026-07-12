"""SwinDocSeg backend."""

from __future__ import annotations

from rare.config.paths import WEIGHTS_PATH
from rare.doc.schema import TAXONOMY_TO_GLASBENA_MLADINA
from rare.models.registry import register

PRED_CATEGORY_MAPS: dict[str, dict[str, str]] = {
    "PubLayNet": {
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

@register("layout", "swindocseg")
class SwinDocSegBackend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        if config is not None:
            self._model = lp.SwinDocSegLayoutModel(**config)
            label_map = config.get("label_map")
        else:
            self._model = lp.SwinDocSegLayoutModel(
                WEIGHTS_PATH + "/swindocseg/model_final_doclay_swindocseg.pth",
                yaml_path="configs/swindocseg/yaml/doclaynet/config_doclay.yaml",
            )
            label_map = "Glasana"
        self.pred_category_map = PRED_CATEGORY_MAPS.get(label_map)
        self.source_taxonomy = (
            label_map if isinstance(label_map, str) and label_map in TAXONOMY_TO_GLASBENA_MLADINA
            else None
        )
        self.label_map = self._model.label_map

    def detect(self, image):
        return self._model.detect(image)
