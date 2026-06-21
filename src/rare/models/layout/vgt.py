"""VGT (Vision Grid Transformer) backend."""

from __future__ import annotations

from rare.config.paths import WEIGHTS_PATH
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
    "D4LA": {
        # headings
        "DocTitle":    "title",
        "ParaTitle":   "title",
        "RegionTitle": "title",
        # body text / lists / misc text regions
        "ParaText":    "text_block",
        "ListText":    "text_block",
        "RegionList":  "text_block",
        "RegionKV":    "text_block",
        "Abstract":    "text_block",
        "Author":      "text_block",
        "Date":        "text_block",
        "Question":    "text_block",
        "OtherText":   "text_block",
        "Catalog":     "text_block",
        "LetterDear":  "text_block",
        "LetterSign":  "text_block",
        "Number":      "text_block",   # ambiguous standalone number; PageNumber covers folios
        # letter / page furniture
        "LetterHead":  "header",
        "PageHeader":  "header",
        "Footer":      "footer",
        "PageFooter":  "footer",
        "PageNumber":  "page_number",
        # figures / tables / equations / refs
        "Figure":      "figure",
        "FigureName":  "figure_caption",
        "Table":       "table",
        "TableName":   "table_caption",
        "Equation":    "equation_isolated",
        "Reference":   "reference",
    }
}

@register("layout", "vgt")
class VGTBackend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        if config is not None:
            self._model = lp.VGTLayoutModel(**config)
            label_map = config.get("label_map")
        else:
            self._model = lp.VGTLayoutModel(
                WEIGHTS_PATH + "/vgt/D4LA_VGT_model.pth",
                grid_root="./data/vgt/grid/",
                yaml_path="configs/vgt/yaml/D4LA_VGT_cascade_PTM.yaml",
            )
            label_map = "Glasana"
        # When predictions already use the Glasbena source vocabulary (the
        # "Glasana" label_map), they share the ground-truth taxonomy, so the
        # runner reuses the GT category map and this stays None.
        self.pred_category_map = PRED_CATEGORY_MAPS.get(label_map)
        self.label_map = self._model.label_map

    def detect(self, image):
        return self._model.detect(image)
