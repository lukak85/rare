"""DocLayout-YOLO backend."""

from __future__ import annotations

from rare.config.paths import WEIGHTS_PATH
from rare.doc.schema import TAXONOMY_TO_GLASBENA_MLADINA
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
    "DocStructBench": {
        "title":           "title",
        "plain text":      "text",
        "abandon":         "abandon",
        "figure":          "figure",
        "figure_caption":  "figure_caption",
        "table":           "table",
        "table_caption":   "table_caption",
        "table_footnote":  "table_footnote",
        "isolate_formula": "equation_isolated",
        "formula_caption": "equation_caption",
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
        "Text":           "text",
        "Title":          "title",
    },
    "D4LA": {
        # headings
        "DocTitle":    "title",
        "ParaTitle":   "title",
        "RegionTitle": "title",
        # body text / lists / misc text regions
        "ParaText":    "text",
        "ListText":    "text",
        "RegionList":  "text",
        "RegionKV":    "text",
        "Abstract":    "text",
        "Author":      "text",
        "Date":        "text",
        "Question":    "text",
        "OtherText":   "text",
        "Catalog":     "text",
        "LetterDear":  "text",
        "LetterSign":  "text",
        "Number":      "text",   # ambiguous standalone number; PageNumber covers folios
        # letter / page furniture
        "LetterHead":  "abandon",
        "PageHeader":  "abandon",
        "Footer":      "abandon",
        "PageFooter":  "abandon",
        "PageNumber":  "abandon",
        # figures / tables / equations / refs
        "Figure":      "figure",
        "FigureName":  "figure_caption",
        "Table":       "table",
        "TableName":   "table_caption",
        "Equation":    "equation_isolated",
        "Reference":   "text",
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
        "paragraph":                   "text",
        "lead":                        "text",
        "byline":                      "text",
        "author":                      "text",
        "translator":                  "text",
        "dateline":                    "text",
        "credit":                      "text",
        "drop cap":                    "text",
        "editor's note":               "text",
        "correction":                  "text",
        "index":                       "text",
        "catalogue":                   "text",
        "institute":                   "text",
        "examinee information":        "text",
        "inside":                      "text",
        "jump line":                   "text",
        "sidebar":                     "text",
        "breakout":                    "text",
        "teasers":                     "text",
        "poem":                        "text",
        "play":                        "text",
        "weather forecast":            "text",
        "bill":                        "text",
        "answer":                      "text",
        "option":                      "text",
        "matching":                    "text",
        "ordered list":                "text",
        "unordered list":              "text",
        "first-level question number": "text",
        "second-level question number":"text",
        "third-level question number": "text",
        "other question number":       "text",
        "supplementary note":          "text",
        # code / formula
        "algorithm":                   "figure",
        "code":                        "figure",
        "formula":                     "isolate_formula",
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
                label_map="DocStructBench",
            )
            label_map = "DocStructBench"
        # When predictions already use the Glasbena source vocabulary (the
        # "Glasana" label_map), they share the ground-truth taxonomy, so the
        # runner reuses the GT category map and this stays None.
        self.pred_category_map = PRED_CATEGORY_MAPS.get(label_map)
        # Advertise the prediction vocabulary to the parse pipeline so it can
        # relabel foreign labels (e.g. D4LA) into Glasbena RegionCategory values
        # before assembly. None when predictions already speak Glasbena
        # ("Glasana" label_map) or when no inbound map exists for the vocabulary.
        self.source_taxonomy = (
            label_map if isinstance(label_map, str) and label_map in TAXONOMY_TO_GLASBENA_MLADINA
            else None
        )
        self.label_map = self._model.label_map

    def detect(self, image):
        return self._model.detect(image)
