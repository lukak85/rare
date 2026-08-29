from __future__ import annotations

import os
from pathlib import Path

from tqdm import tqdm

from rare.doc.schema import GlasanaDocument
from rare.models.registry import register

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@register("vlm", "youtu")
class YoutuBackend:

    name = "youtu"

    def __init__(self, config: dict | None = None):
        self._converter = None  # built lazily on first use
        cfg = config or {}
        # Read rather than required: a document whose pages are already parsed
        # is rebuilt from the JSON on disk, and never reaches the model.
        self.model_path = cfg.get("model_path")
        self.angle_correct_model_path = cfg.get("angle_correct_model_path")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Force everything on the same device
        self.verbose = True
        # parse_pdf: DPI the PDF is rendered at, and where the parser's own
        # JSON/Markdown/preview files are kept. Defaulting the work directory
        # to a path rather than a temp dir is deliberate — a Youtu run is slow
        # enough that being able to re-render the document from JSON, without
        # re-running the model, is worth more than tidiness.
        self.dpi = int(cfg.get("dpi", 200))
        self.work_dir = cfg.get("work_dir", "outputs/youtu")

    def _get_converter(self):
        if self._converter is None:
            if not self.model_path:
                raise RuntimeError(
                    "youtu: no model_path in the backend config, and a page "
                    "still has to be parsed. Pass --config configs/youtu/config.json."
                )
            from youtu_hf_parser import YoutuOCRParserHF
            self._converter = YoutuOCRParserHF(
                model_path=self.model_path,                    # Path to downloaded model weights
                enable_angle_correct=True,                # Set to False to disable angle correction
                angle_correct_model_path=self.angle_correct_model_path  # If None, model will auto-download to default path; if custom path, manually download https://github.com/TencentCloudADP/youtu-parsing/releases/download/v1.0.0/model.pth to specified location
            )
        return self._converter

    @staticmethod
    def _load_image_paths(image_dir: str | Path) -> list[str]:
        """Recursively collect supported images under `image_dir`, returning
        parallel lists of absolute paths and RGB PIL images."""
        image_paths: list[str] = []
        for root, _dirs, files in os.walk(image_dir):
            for file in files:
                if os.path.splitext(file.lower())[1] in SUPPORTED_EXTENSIONS:
                    image_paths.append(os.path.abspath(os.path.join(root, file)))
        image_paths.sort()
        print(f"found {len(image_paths)} image files.")
        return image_paths

    def to_markdown(
        self,
        pdf_dir: str | Path,
        image_dir: str | Path,
        out_md_dir: str | Path,
        skip_existing: bool = False,
    ) -> str | Path:
        image_paths = self._load_image_paths(image_dir)

        for image_path in tqdm(image_paths):
            out_md = Path(out_md_dir) / f"{Path(image_path).stem}.md"   # xx_1.jpg -> xx_1.md
            if skip_existing and out_md.exists():
                print(f"skipping {out_md}")
                continue
            try:
                self._get_converter().parse_file(
                    input_path=image_path,     # Input document path
                    output_dir=out_md_dir      # Output directory for results
                )
            except Exception as e:
                if self.verbose:
                    print(e)
                try:
                    import re
                    lower_res_image_path = re.sub(r"/eval_\d+dpi/", "/eval/", image_path)
                    print(f"Using {lower_res_image_path}")
                    self._get_converter().parse_file(
                        input_path=lower_res_image_path,     # Input document path
                        output_dir=out_md_dir      # Output directory for results
                    )
                except Exception as e2:
                    if self.verbose:
                        print(e2)
                    print("Skipping " + image_path)
                    pass

        return out_md_dir

    # --- evaluate/runner.py::run_vlm contract ------------------------------

    def parse_pdf(self, pdf_path: str | Path) -> GlasanaDocument:
        """Render `pdf_path`, parse every page, and rebuild a GlasanaDocument.

        Page images are written to `<work_dir>/images/<stem>_<page>.jpg` and the
        parser's output to `<work_dir>/<stem>/`, both kept afterwards. That is
        what makes this the only Youtu entry point needed: a page whose JSON is
        already there is not parsed again, so re-running the same command on a
        finished document rebuilds it from disk — no GPU, no model weights in
        the config, and not even a PDF render, since the page images are kept
        too. Pages the parser fails on are skipped with a warning rather than
        aborting the document.
        """
        from rare.parse.pdf import page_count, render_pages
        from rare.parse.youtu import collect_pages, document_from_pages

        pdf_path = Path(pdf_path)
        stem = pdf_path.stem
        work_dir = Path(self.work_dir) / stem
        images_dir = Path(self.work_dir) / "images"
        work_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        # A page needs the model when its JSON is missing, and needs a render
        # when its image is missing — the crops in `document_from_pages` come
        # off that image, so a re-render still wants it on disk.
        todo = [
            page_no
            for page_no in range(page_count(pdf_path))
            if not (work_dir / f"{stem}_{page_no}.json").exists()
            or not (images_dir / f"{stem}_{page_no}.jpg").exists()
        ]
        if todo:
            page_images = render_pages(pdf_path, dpi=self.dpi)
            # Loaded once, before the loop: a missing model path or missing
            # weights is the whole run's problem, not one page's, and inside
            # the loop it would come out as a per-page warning and a document
            # quietly short of its pages.
            unparsed = [p for p in todo if not (work_dir / f"{stem}_{p}.json").exists()]
            converter = self._get_converter() if unparsed else None
            for page_no in tqdm(todo):
                image_path = images_dir / f"{stem}_{page_no}.jpg"
                if not image_path.exists():
                    page_images[page_no].convert("RGB").save(image_path)
                if page_no not in unparsed:
                    continue
                try:
                    converter.parse_file(
                        input_path=str(image_path), output_dir=str(work_dir)
                    )
                except Exception as exc:
                    print(f"[warn] Youtu failed on page {page_no} of {stem}: {exc}")

        pages = collect_pages(work_dir).get(stem, [])
        if not pages:
            print(f"[warn] no Youtu page JSON under {work_dir}")
        return document_from_pages(
            stem, pages, images_dir=images_dir, figures_dir=work_dir / "figures"
        )
