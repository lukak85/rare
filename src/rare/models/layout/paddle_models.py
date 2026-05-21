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

    def detect(self, image):
        return self._model.detect(image)
