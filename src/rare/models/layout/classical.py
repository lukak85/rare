"""Classical (non-learned) layout backends: Docstrum, RLSA, Recursive-XYCut."""

from __future__ import annotations

from rare.models.registry import register


@register("layout", "docstrum")
class DocstrumBackend:
    def __init__(self, config: dict | None = None, verbose: bool = False):
        import layoutparser as lp
        self._model = (
            lp.DocstrumLayoutModel(**config, verbose=verbose)
            if config
            else lp.DocstrumLayoutModel(verbose=verbose)
        )

    def detect(self, image):
        return self._model.detect(image)


@register("layout", "rlsa")
class RLSABackend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        self._model = lp.RLSALayoutModel(**config) if config else lp.RLSALayoutModel()

    def detect(self, image):
        return self._model.detect(image)


@register("layout", "recursive-xycut")
class RecursiveXYCutBackend:
    def __init__(self, config: dict | None = None):
        import layoutparser as lp
        self._model = (
            lp.RecursiveXYCutLayoutModel(**config)
            if config
            else lp.RecursiveXYCutLayoutModel()
        )

    def detect(self, image):
        return self._model.detect(image)
