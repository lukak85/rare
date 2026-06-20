"""DocLayout-YOLO backend."""

from __future__ import annotations

from rare.config.paths import WEIGHTS_PATH
from rare.models.registry import register


OMNIDOCBENCH_PRED_CAT_MAPPING ="""\
title : title
plain text: text
abandon: abandon
figure: figure
figure_caption: figure_caption
"""

# Per-`label_map` translation of the model's predicted category names into the
# shared OmniDocBench `category_type` space, so predictions from a detector
# trained on a foreign taxonomy can be scored against Glasbena ground truth
# (see `mean_average_precision` / `score_layout`). Names absent from a map fall
# through unchanged. Targets match the values in
# `rare.evaluate.omnidocbench.DEFAULT_CATEGORY_MAP`.
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
    },
    "DocSynth300K": {
        # headings (all title levels collapse to OmniDocBench `title`)
        "title":                       "title",
        "headline":                    "title",
        "chapter title":               "title",
        "section":                     "title",
        "section title":               "title",
        "sub section title":           "title",
        "subsub section title":        "title",
        "subhead":                     "title",
        "kicker":                      "title",
        "part":                        "title",
        "first-level title":           "title",
        "second-level title":          "title",
        "third-level title":           "title",
        "fourth-level title":          "title",
        "fourth-level section title":  "title",
        "fifth-level title":           "title",
        # body / list / generic text regions
        "paragraph":                   "text_block",
        "lead":                        "text_block",
        "byline":                      "text_block",
        "author":                      "text_block",
        "translator":                  "text_block",
        "dateline":                    "text_block",
        "credit":                      "text_block",
        "drop cap":                    "text_block",
        "editor's note":               "text_block",
        "correction":                  "text_block",
        "index":                       "text_block",
        "catalogue":                   "text_block",
        "institute":                   "text_block",
        "examinee information":        "text_block",
        "inside":                      "text_block",
        "jump line":                   "text_block",
        "sidebar":                     "text_block",
        "breakout":                    "text_block",
        "teasers":                     "text_block",
        "poem":                        "text_block",
        "play":                        "text_block",
        "weather forecast":            "text_block",
        "bill":                        "text_block",
        "answer":                      "text_block",
        "option":                      "text_block",
        "matching":                    "text_block",
        "ordered list":                "text_block",
        "unordered list":              "text_block",
        "first-level question number": "text_block",
        "second-level question number":"text_block",
        "third-level question number": "text_block",
        "other question number":       "text_block",
        "supplementary note":          "text_block",
        # code / formula
        "algorithm":                   "code_txt",
        "code":                        "code_txt",
        "formula":                     "equation_isolated",
        # captions / footnotes / references
        "caption":                     "figure_caption",
        "table caption":               "table_caption",
        "table note":                  "table_footnote",
        "footnote":                    "page_footnote",
        "endnote":                     "page_footnote",
        "marginal note":               "page_footnote",
        "reference":                   "reference",
        # figures / graphical regions / tables
        "figure":                      "figure",
        "mugshot":                     "figure",
        "QR code":                     "figure",
        "barcode":                     "figure",
        "table":                       "table",
        # page furniture
        "header":                      "header",
        "footer":                      "footer",
        "page number":                 "page_number",
        "folio":                       "page_number",
        # non-content / furniture -> dropped from scoring
        "advertisement":               "abandon",
        "blank":                       "abandon",
        "bracket":                     "abandon",
        "flag":                        "abandon",
        "sealing line":                "abandon",
        "underscore":                  "abandon",
    },
}


@register("layout", "doclayout-yolo")
class DocLayoutYOLOBackend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        if config is not None:
            self._model = lp.DocLayoutYOLOLayoutModel(**config)
            label_map = config.get("label_map")
        else:
            self._model = lp.DocLayoutYOLOLayoutModel(
                WEIGHTS_PATH + "/model/doclayoutyolo/doclayout_yolo_docstructbench_imgsz1024.pt",
                label_map="Glasana",
            )
            label_map = "Glasana"
        # When predictions already use the Glasbena source vocabulary (the
        # "Glasana" label_map), they share the ground-truth taxonomy, so the
        # runner reuses the GT category map and this stays None.
        self.pred_category_map = PRED_CATEGORY_MAPS.get(label_map)
        self.label_map = self._model.label_map

    def detect(self, image):
        return self._model.detect(image)
