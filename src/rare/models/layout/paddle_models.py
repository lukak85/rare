"""PaddlePaddle-backed layout models: ppyolo, pp-doclayoutv3."""

from __future__ import annotations

from rare.models.registry import register


_PUBLAYNET_LABEL_MAP = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}


@register("layout", "ppyolo")
class PPYOLOBackend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        if config is not None:
            self._model = lp.PaddleDetectionLayoutModel(
                **config, label_map=_PUBLAYNET_LABEL_MAP
            )
        else:
            self._model = lp.PaddleDetectionLayoutModel(
                config_path="./data/model/ppyolo/publaynet/ppyolov2_r50vd_dcn_365e_publaynet/infer_cfg.yml",
                model_path="./data/model/ppyolo/publaynet/ppyolov2_r50vd_dcn_365e_publaynet",
                label_map=_PUBLAYNET_LABEL_MAP,
            )

    def detect(self, image):
        return self._model.detect(image)


@register("layout", "pp-doclayoutv3")
class PPDocLayoutV3Backend:
    """PP-DocLayoutV3 returns regions already in reading order; pair with
    `--order pp-doclayoutv3-builtin` to skip a separate ordering step."""

    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        self._model = (
            lp.PPDocLayoutV3LayoutModel(**config) if config else lp.PPDocLayoutV3LayoutModel()
        )
        if config is not None:
            label_map = config.get("label_map")
        else:
            label_map = "Glasana"
        # When predictions already use the Glasbena source vocabulary (the
        # "Glasana" label_map), they share the ground-truth taxonomy, so the
        # runner reuses the GT category map and this stays None.

        # Classes were inferred from: https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L/blob/main/config.json#L39
        self.pred_category_map = {
            "paragraph_title": "title",
            "image": "figure",
            "text": "text_block",
            "number": "page_number",
            "abstract": "text_block",
            "content": "text_block",
            "figure_title": "figure_caption",
            "formula": "equation_isolated",
            "table": "table",
            "reference": "reference",
            "doc_title": "title",
            "footnote": "page_footnote",
            "header": "header",
            "algorithm": "code_txt",
            "footer": "footer",
            "seal": "abandon",
            "chart": "figure",
            "formula_number": "figure_caption",
            "aside_text": "text_block",
            "reference_content": "text_block"
        }
        self.label_map = {
            0: "paragraph_title",
            1: "image",
            2: "text",
            3: "number",
            4: "abstract",
            5: "content",
            6: "figure_title",
            7: "formula",
            8: "table",
            9: "reference",
            10: "doc_title",
            11: "footnote",
            12: "header",
            13: "algorithm",
            14: "footer",
            15: "seal",
            16: "chart",
            17: "formula_number",
            18: "aside_text",
            19: "reference_content"
        }

    def detect(self, image):
        return self._model.detect(image)
