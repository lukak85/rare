"""Unified command-line interface: `rare parse | evaluate | tools`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from rare.models.registry import ensure_layoutparser_backend, get, list_backends


ALL_TEXT_LABELS = ",".join((
    "Header", "Footer", "PageNum", "Section", "Dateline", "EditNote",
    "MarginNote",
    "Headline", "Kicker", "Deck", "Subhead", "Subsubhead", "Author", "Byline",
    "Translator",
    "Paragraph", "Quote", "Dropcap",
    "Caption", "FigByline",
    "Table", "OrderedList", "UnorderedList",
    "Footnote", "TOC", "Literary", "Literature",
    "Question",
))


def _read_config(path: str | None) -> dict | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text())


def _make_backend(kind: str, name: str, config_path: str | None = None, **extra):
    """Instantiate a registry backend, passing a config only if it takes one.

    `extra` supplies defaults the CLI can infer (e.g. `pdf_root` from the PDF
    already named on the command line, so line-level ordering works without
    repeating the path in a config file). Explicit config keys always win.
    """
    import inspect

    cls = get(kind, name)
    if "config" not in inspect.signature(cls.__init__).parameters:
        return cls()

    cfg = dict(_read_config(config_path) or {})
    for key, value in extra.items():
        if value and not cfg.get(key):
            cfg[key] = str(value)
    return cls(config=cfg)


def _make_order(name: str, config_path: str | None = None, pdf_root=None):
    """Instantiate a reading-order backend."""
    return _make_backend("order", name, config_path, pdf_root=pdf_root)


def _make_ner(args: argparse.Namespace):
    """Instantiate the NER backend for the linking stage, or None.

    Linking is best-effort: the model extras are installed per backend, so a
    missing `transformers`/`torch` must degrade the result (geometry-only
    linking) rather than fail an otherwise successful parse.
    """
    if not getattr(args, "ner", None):
        return None
    try:
        return _make_backend("ner", args.ner, None)
    except Exception as exc:  # noqa: BLE001 — any import/init failure is non-fatal
        print(
            f"warning: NER backend '{args.ner}' unavailable ({exc}); "
            "linking will run without named entities.",
            file=sys.stderr,
        )
        return None


def _make_classifier(args: argparse.Namespace):
    """Instantiate the article-genre backend for the linking stage, or None.

    Best-effort for the same reason as `_make_ner`, and more so: these are
    large generative models whose weights may simply not be present.
    """
    name = getattr(args, "classification", None)
    if not name or name.lower() == "none":
        return None
    try:
        return _make_backend("classification", name, getattr(args, "classification_config", None))
    except Exception as exc:  # noqa: BLE001 — any import/init failure is non-fatal
        print(
            f"warning: classification backend '{name}' unavailable ({exc}); "
            "articles will be left unclassified.",
            file=sys.stderr,
        )
        return None


OCR_BACKENDS = ("tesseract", "ppocr")


def _make_ocr(args: argparse.Namespace):
    """Instantiate the OCR fallback, or None.

    Not best-effort, unlike `_make_ner`: --ocr was asked for explicitly, and a
    missing language pack would otherwise fall through to English and fill the
    document with plausible-looking text whose diacritics are wrong.

    Naming more than one backend wraps them in `BestOfOCR`, which reads every
    region with each and keeps the better answer — see `rare.parse.ocr`.
    """
    requested = [b.strip() for b in getattr(args, "ocr", "").split(",") if b.strip()]
    if not requested:
        return None

    unknown = [b for b in requested if b not in OCR_BACKENDS]
    if unknown:
        raise SystemExit(
            f"error: unknown --ocr backend(s): {', '.join(unknown)} "
            f"(choose from {', '.join(OCR_BACKENDS)})"
        )

    built = []
    for backend in dict.fromkeys(requested):        # de-duplicated, order kept
        if backend == "tesseract":
            from rare.parse.ocr import TesseractOCR

            built.append(TesseractOCR(
                lang=args.ocr_lang,
                dpi=args.ocr_dpi,
                min_confidence=args.ocr_min_confidence,
                min_replace_confidence=args.ocr_min_replace_confidence,
            ))
        else:
            from rare.parse.ocr_ppocr import make_recognizer

            built.append(make_recognizer(
                rec_model=args.ocr_rec_model,
                dpi=args.ocr_dpi,
                min_confidence=args.ocr_min_confidence,
                min_replace_confidence=args.ocr_min_replace_confidence,
                solidify=args.ocr_fill_outlines,
            ))

    if len(built) == 1:
        return built[0]

    from rare.parse.ocr import BestOfOCR

    return BestOfOCR(built)


def _add_ocr_flags(parser: argparse.ArgumentParser) -> None:
    """The OCR fallback's flags, shared by `parse` and `evaluate`.

    Identical on both so a parse run and the evaluation of that same
    configuration are spelt the same way. On the evaluate side they apply to
    the predicted text only — see `rare.evaluate.pdf_text`.
    """
    parser.add_argument(
        "--ocr",
        default="",
        metavar="BACKEND[,BACKEND]",
        help="Re-read regions the PDF's text layer left empty by OCR'ing the "
             "rendered page (see rare.parse.ocr). One of tesseract, ppocr, or "
             "both comma-separated — the two fail in opposite directions on "
             "letterspaced display type, and naming both reads every region "
             "with each and keeps the better answer. Off by default.",
    )
    parser.add_argument(
        "--ocr-labels",
        default=ALL_TEXT_LABELS,
        metavar="LABELS",
        help="Comma-separated labels the OCR fallback may fill. Defaults to "
             "every text-bearing region in the annotation scheme; pass a "
             "shorter list (e.g. Header,Kicker) to narrow it.",
    )
    parser.add_argument(
        "--ocr-retry",
        nargs="?",
        const="junk,sparse,alien,empty",
        default="",
        metavar="REASONS",
        help="Also re-read regions whose text is present but looks broken, as "
             "scored by rare.parse.quality: junk (tokens that are not words, "
             "e.g. the headline 'W Z7'), sparse (far less text than a box that "
             "shape holds), alien (characters outside the Slovene alphabet). "
             "Bare --ocr-retry means all three. Off by default, in which case "
             "only empty regions are filled. Replacing existing text needs "
             "--ocr-min-replace-confidence and a reading that scores clean, so "
             "one bad reading is never swapped for another.",
    )
    parser.add_argument(
        "--ocr-min-replace-confidence",
        type=float,
        default=60.0,
        help="Confidence an OCR reading needs before it may overwrite text the "
             "PDF already carried (default: 60, above --ocr-min-confidence on "
             "purpose).",
    )
    parser.add_argument(
        "--ocr-rec-model",
        default="latin_PP-OCRv5_mobile_rec",
        help="Recognition model for --ocr ppocr (default: "
             "latin_PP-OCRv5_mobile_rec, the Latin-script multilingual head). "
             "The Chinese/English heads have no č/š/ž in their dictionary and "
             "transliterate them away silently.",
    )
    parser.add_argument(
        "--ocr-fill-outlines",
        action="store_true",
        help="For --ocr ppocr: solidify hollow letterforms before recognition. "
             "Display headlines here are sometimes set in an outline face, "
             "where only the contour is inked and every recogniser sees rings. "
             "Off by default — it trades one error for another on filled type.",
    )
    parser.add_argument(
        "--ocr-lang",
        default="slv",
        help="Tesseract language for --ocr (default: slv). Fails if not installed.",
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=400,
        help="DPI the OCR crops are rendered at (default: 400, above --dpi on "
             "purpose: 200 is marginal for Tesseract on these scans).",
    )
    parser.add_argument(
        "--ocr-min-confidence",
        type=float,
        default=40.0,
        help="Discard OCR readings below this mean word confidence (default: 40).",
    )


def _ocr_labels(args: argparse.Namespace) -> list[str]:
    return [label.strip() for label in args.ocr_labels.split(",") if label.strip()]


def _ocr_retry(args: argparse.Namespace) -> list[str]:
    """Quality reasons that earn a *non-empty* region a second reading.

    Validated here rather than in `fill_failed_regions`: a misspelt reason is
    silently no retry at all, and the run would look like the flag worked.
    """
    from rare.parse.quality import REASONS

    reasons = [r.strip() for r in getattr(args, "ocr_retry", "").split(",") if r.strip()]
    unknown = set(reasons) - set(REASONS)
    if unknown:
        raise SystemExit(
            f"error: unknown --ocr-retry reason(s): {', '.join(sorted(unknown))} "
            f"(choose from {', '.join(r for r in REASONS if r != 'empty')})"
        )
    # --ocr-retry only widens which regions the OCR fallback is offered; with no
    # backend there is nothing to offer them to, and the run would otherwise
    # finish looking like the flag had worked.
    if reasons and not [b for b in getattr(args, "ocr", "").split(",") if b.strip()]:
        raise SystemExit(
            "error: --ocr-retry needs a backend. It widens which regions get "
            "re-read; --ocr chooses what re-reads them. Add --ocr tesseract "
            "(or ppocr, or tesseract,ppocr)."
        )
    return reasons


def _link(doc, args: argparse.Namespace):
    """Run the whole-document linking passes unless --no-link was given."""
    from rare.link import link_document

    return link_document(
        doc,
        ner=_make_ner(args),
        classifier=_make_classifier(args),
        config=_read_config(getattr(args, "link_config", None)),
    )


def cmd_parse(args: argparse.Namespace) -> int:
    if args.list_models:
        print("Layout backends:")
        for n in list_backends("layout"):
            print(f"  - {n}")
        print("\nReading-order backends:")
        for n in list_backends("order"):
            print(f"  - {n}")
        print("\nVLM backends:")
        for n in list_backends("vlm"):
            print(f"  - {n}")
        print("\nNER backends (linking stage):")
        for n in list_backends("ner"):
            print(f"  - {n}")
        print("\nClassification backends:")
        for n in list_backends("classification"):
            print(f"  - {n}")
        return 0

    # COCO track — render an existing COCO layout (no detection step).
    if args.coco:
        if args.layout or args.vlm:
            print("error: --coco cannot be combined with --layout or --vlm.", file=sys.stderr)
            return 2
        from rare.parse.coco import parse_coco

        # Only build a reading-order backend when one is explicitly requested;
        # otherwise the COCO `order_id` field (then top-bottom) is used directly.
        order = None
        if args.order and args.order != "top-bottom":
            order = _make_order(
                args.order, args.order_config,
                pdf_root=args.pdfs_dir or (Path(args.pdf).parent if args.pdf else None),
            )

        category_map = None
        if args.category_map:
            from rare.evaluate.omnidocbench import load_category_map
            category_map = load_category_map(args.category_map)

        out_dirs = parse_coco(
            args.coco,
            pdf_path=args.pdf,
            images_dir=args.images_dir,
            pdfs_dir=args.pdfs_dir,
            order=order,
            output_dir=args.output,
            dpi=args.dpi,
            emit_omnidocbench=args.emit_omnidocbench,
            category_map=category_map,
            linker=None if args.no_link else (lambda doc: _link(doc, args)),
            ocr=_make_ocr(args),
            ocr_labels=_ocr_labels(args),
            ocr_retry=_ocr_retry(args),
        )
        for out in out_dirs:
            print(f"Output written to: {out}")
        if args.emit_omnidocbench:
            print(f"OmniDocBench JSON written to: {Path(args.output) / 'omnidocbench.json'}")
        return 0

    if not args.pdf:
        print(
            "error: missing PDF path. Usage:\n"
            "  rare parse <pdf> --layout <name> [--order <name>]\n"
            "  rare parse <pdf> --vlm <name>\n"
            "  rare parse --coco <coco.json> [--images-dir <dir>] [--pdfs-dir <dir>]",
            file=sys.stderr,
        )
        return 2

    if args.vlm and args.layout:
        print("error: pass either --vlm or --layout, not both.", file=sys.stderr)
        return 2
    if not args.vlm and not args.layout:
        print("error: one of --layout, --vlm, or --coco is required.", file=sys.stderr)
        return 2

    if args.vlm:
        # VLM track — produces a GlasanaDocument directly.
        from rare.parse.io import write_outputs

        vlm_cls = get("vlm", args.vlm)
        vlm = vlm_cls(config=_read_config(args.config))
        doc = vlm.parse_pdf(args.pdf)
        if not args.no_link:
            _link(doc, args)
        out = write_outputs(doc, args.output, per_page=args.per_page)
        print(f"Output written to: {out}")
        return 0

    # Pipeline track. Instantiating the layout backend first ensures
    # LAYOUTPARSER_BACKEND is set before any layoutparser import.
    layout_cls = get("layout", args.layout)
    layout = layout_cls(config=_read_config(args.config))

    # `pdf_root` is the parsed PDF's own directory: line-level orderers resolve
    # `<pdf_root>/<pdf_stem>.pdf`, and pipeline.parse_pdf passes that same stem.
    order = _make_order(
        args.order, args.order_config,
        pdf_root=Path(args.pdf).parent if args.pdf else None,
    )

    # Imported only after the layout backend is constructed.
    from rare.parse.pipeline import parse_pdf

    out = parse_pdf(
        args.pdf, layout, order,
        output_dir=args.output,
        dpi=args.dpi,
        per_page=args.per_page,
        save_coco=args.emit_coco,
        emit_omnidocbench=args.emit_omnidocbench,
        linker=(lambda doc: _link(doc, args)),
        ocr=_make_ocr(args),
        ocr_labels=_ocr_labels(args),
        ocr_retry=_ocr_retry(args),
    )
    print(f"Output written to: {out}")
    if args.emit_coco:
        print(f"COCO JSON written to: {out / f'{Path(args.pdf).stem}_coco.json'}")
    if args.emit_omnidocbench:
        print(f"OmniDocBench per-page Markdown written to: {out / 'omnidocbench'}")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    # Stub — never reached. `tools` is dispatched in main() before argparse runs
    # so the helper can own its own flag namespace.
    return 0


def _resolve_annotations(args: argparse.Namespace) -> Path | None:
    """The COCO file the document-level tracks score against, or None."""
    root = Path(args.data_root or f"datasets/{args.dataset}")
    if getattr(args, "annotations", None):
        path = Path(args.annotations)
        return path if path.exists() else None
    if (root / "annotations.json").exists():
        return root / "annotations.json"
    return None


def _evaluate_page_genre(args: argparse.Namespace) -> int:
    """`--track page-genre`: annotated page type vs predicted genre (see rare.evaluate.page_genre)."""
    from rare.evaluate.page_genre import run_page_genre

    root = Path(args.data_root or f"datasets/{args.dataset}")
    coco_path = _resolve_annotations(args)
    if coco_path is None:
        print(f"error: no COCO annotations found under {root}.", file=sys.stderr)
        return 2

    pdfs_dir = Path(args.pdfs_dir) if args.pdfs_dir else root / "pdfs"
    run_id = args.run_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output) / run_id

    summary = run_page_genre(
        coco_path,
        run_dir,
        docs_dir=args.docs_dir,
        pdfs_dir=pdfs_dir if pdfs_dir.exists() else None,
        linker=(lambda doc: _link(doc, args)) if not args.docs_dir else None,
        page_type_map=args.page_type_map,
        limit=args.limit,
        dataset_name=args.dataset,
    )
    print(f"\nAggregates: {json.dumps(summary['overall'], indent=2)}")
    print(f"Scored page types: {', '.join(summary['scored_page_types'])}")
    print(f"Ignored page types: {', '.join(summary['ignored_page_types'])}")
    print(f"Report: {run_dir / 'report.md'}")
    return 0


def _evaluate_figure_link(args: argparse.Namespace) -> int:
    """`--track figure-link`: figure → article placement (see rare.evaluate.figure_link).

    Currently scores parsed documents rather than building them.
    """
    from rare.evaluate.figure_link import run_figure_link

    docs_dir = Path(args.docs_dir or "outputs/parsed/gt")
    if not docs_dir.exists():
        print(f"error: no parsed documents at {docs_dir}. Build them with\n"
              f"  rare parse --coco <annotations.json> --output outputs/parsed/gt",
              file=sys.stderr)
        return 2

    # Optional: only the annotated `page_type` is read, for the breakdown.
    coco_path = _resolve_annotations(args)
    run_id = args.run_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output) / run_id

    try:
        summary = run_figure_link(
            run_dir,
            docs_dir,
            coco_path=coco_path,
            config=_read_config(getattr(args, "link_config", None)),
            variants=[v for v in (args.variants or "").split(",") if v] or None,
            limit=args.limit,
            dataset_name=args.dataset,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"\nAggregates: {json.dumps(summary['overall'], indent=2)}")
    print("\nBy variant:")
    for name, scores in summary["variants"].items():
        print(f"  {name:<10} accuracy {scores['accuracy']:.4f}"
              f"  (coverage {scores['coverage']:.3f})")
    print(f"\nReport: {run_dir / 'report.md'}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    if args.list_models:
        return cmd_parse(args)  # reuse the same listing

    # The document-level tracks read the COCO annotations directly — they need
    # no gold layouts, so they skip the loader (and layoutparser with it).
    if args.track == "figure-link":
        return _evaluate_figure_link(args)
    if args.track == "page-genre":
        return _evaluate_page_genre(args)

    # Must run before dataset loading: gold-layout construction imports
    # layoutparser, which freezes LAYOUTPARSER_BACKEND for the process.
    if args.track == "pipeline" and args.layout:
        ensure_layoutparser_backend(args.layout)

    from rare.evaluate import datasets as ds_loader

    # Only forward kwargs the chosen loader actually accepts; loaders differ
    # (e.g. omnidocbench takes no `pdfs_dir`, doclaynet/publaynet take neither
    # `images_dir` nor `pdfs_dir`). The remaining values still reach the runner.
    import inspect
    loader_params = inspect.signature(ds_loader.DATASETS[args.dataset]).parameters
    candidate_kwargs = {
        "root":       args.data_root,
        "pdfs_dir":   args.pdfs_dir,
        "images_dir": args.images_dir,
    }
    ds_loader_kwargs = {
        k: v for k, v in candidate_kwargs.items() if v and k in loader_params
    }
    dataset = ds_loader.load(args.dataset, **ds_loader_kwargs)

    run_id = args.run_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.track == "pipeline":
        if not args.layout:
            print("error: --layout required for --track pipeline.", file=sys.stderr)
            return 2
        layout_cls = get("layout", args.layout)
        layout = layout_cls(config=_read_config(args.config))

        from rare.evaluate.runner import run_pipeline, _resolve_pdfs_dir
        from rare.evaluate.omnidocbench import load_category_map
        images_dir = Path(args.images_dir) if args.images_dir else None
        category_map = load_category_map(args.category_map) if args.category_map else None
        pdfs_dir = Path(args.pdfs_dir) if args.pdfs_dir else None

        # Same PDF directory the OmniDocBench text path uses, so a line-level
        # order backend finds the source PDFs without extra configuration.
        order = _make_order(
            args.order, args.order_config,
            pdf_root=_resolve_pdfs_dir(pdfs_dir, dataset),
        )

        # --omnidocbench-eval only *selects* between passes that
        # --run-omnidocbench turns on. Naming one without it is how you get a
        # run with no prediction markdown and no explanation of why.
        if args.omnidocbench_eval and not args.run_omnidocbench:
            print(
                f"error: --omnidocbench-eval {args.omnidocbench_eval} does "
                f"nothing without --run-omnidocbench, which is what turns the "
                f"OmniDocBench passes on. Add it, or drop this flag.",
                file=sys.stderr,
            )
            return 2

        emit_omnidocbench = args.emit_omnidocbench
        agg = run_pipeline(
            dataset, layout, order, run_dir,
            limit=args.limit,
            start=args.start,
            emit_omnidocbench=emit_omnidocbench,
            category_map=category_map,
            pdfs_dir=pdfs_dir,
            run_omnidocbench=args.run_omnidocbench,
            omnidocbench_image=args.omnidocbench_image,
            omnidocbench_layout_image=args.omnidocbench_layout_image,
            omnidocbench_ground=args.omnidocbench_ground,
            omnidocbench_eval=args.omnidocbench_eval or "both",
            ocr=_make_ocr(args),
            ocr_labels=_ocr_labels(args),
            ocr_retry=_ocr_retry(args),
        )

    elif args.track == "vlm":
        if not args.vlm:
            print("error: --vlm required for --track vlm.", file=sys.stderr)
            return 2
        vlm_cls = get("vlm", args.vlm)
        vlm = vlm_cls(config=_read_config(args.config))

        from rare.evaluate.runner import run_vlm
        from rare.evaluate.omnidocbench import load_category_map
        pdfs_dir = Path(args.pdfs_dir) if args.pdfs_dir else None
        images_dir = Path(args.images_dir) if args.images_dir else None
        category_map = load_category_map(args.category_map) if args.category_map else None
        agg = run_vlm(
            dataset, vlm, run_dir,
            limit=args.limit,
            pdfs_dir=pdfs_dir,
            images_dir=images_dir,
            category_map=category_map,
            run_omnidocbench=args.run_omnidocbench,
            omnidocbench_image=args.omnidocbench_image,
            omnidocbench_ground=args.omnidocbench_ground
        )

    else:
        print(f"error: unknown --track '{args.track}'.", file=sys.stderr)
        return 2

    print(f"\nAggregates: {json.dumps(agg, indent=2)}")
    print(f"Report: {run_dir / 'report.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rare",
        description="Slovene-magazine PDF parser and DLA/VLM model-comparison toolkit.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_parse = sub.add_parser("parse", help="Parse a PDF into HTML / Markdown / JSON.")
    p_parse.add_argument("pdf", nargs="?", help="Path to the input PDF.")
    p_parse.add_argument(
        "--layout",
        help="Pipeline-track layout backend (see --list-models).",
    )
    p_parse.add_argument(
        "--order",
        default="top-bottom",
        help="Pipeline-track reading-order backend (default: top-bottom).",
    )
    p_parse.add_argument(
        "--vlm",
        help="VLM-track backend (see --list-models). Mutually exclusive with --layout.",
    )
    p_parse.add_argument(
        "--coco",
        help="COCO-track: render an existing COCO layout JSON (ground truth or any "
             "predictions) to HTML/MD/JSON, skipping detection. Mutually exclusive "
             "with --layout/--vlm. Reading order uses per-annotation `order_id` when "
             "present, then --order, then top-bottom.",
    )
    p_parse.add_argument(
        "--images-dir",
        help="COCO-track: directory of page images (for figure crops).",
    )
    p_parse.add_argument(
        "--pdfs-dir",
        help="COCO-track: directory of source PDFs (<stem>.pdf) used to fill region "
             "text via pdfplumber. Without it, regions render with empty text.",
    )
    p_parse.add_argument(
        "--per-page",
        action="store_true",
        help="Output Markdown in per-page format.",
    )
    p_parse.add_argument(
        "--ner",
        default="rudar-slv",
        help="NER backend used by the linking stage. Default: rudar-slv (Slovenian).",
    )
    p_parse.add_argument(
        "--classification",
        default="gams",
        help="Backend that tags each article with an editorial genre "
             "(recenzija, novica, intervju, ...). Default: gams. Use "
             "'--classification none' to skip it — gams loads a 12B model.",
    )
    p_parse.add_argument(
        "--classification-config",
        dest="classification_config",
        help="JSON config for the classification backend.",
    )
    p_parse.add_argument(
        "--no-link",
        dest="no_link",
        action="store_true",
        help="Skip the whole-document linking passes (captions, articles, "
             "continuations, entity edges); items keep the per-page assembly only.",
    )
    p_parse.add_argument(
        "--emit-omnidocbench",
        dest="emit_omnidocbench",
        action="store_true",
        help="COCO-track: also write <output>/omnidocbench.json (per-page list with "
             "per-region text from the PDF) for evaluating VLMs against OmniDocBench. "
             "Pipeline-track: also write one Markdown file per page to "
             "<output>/<stem>/omnidocbench/<stem>_<page>.md, rendered from the "
             "regions as detected (before paragraphs are merged), for "
             "OmniDocBench's end2end evaluator.",
    )
    p_parse.add_argument(
        "--emit-coco",
        dest="emit_coco",
        action="store_true",
        help="Write resulting COCO JSON to a file.",
    )
    p_parse.add_argument(
        "--category-map",
        help="COCO-track: JSON file of {source_category_name: omnidocbench_category_type} "
             "merged on top of the built-in default map (used by --emit-omnidocbench).",
    )
    p_parse.add_argument(
        "--config",
        help="JSON config file passed to the chosen backend.",
    )
    p_parse.add_argument(
        "--order-config",
        help="JSON config file for the reading-order backend (e.g. layoutreader's "
             "pdf_root / granularity / overlap_thresh). Backends that take no "
             "config ignore it.",
    )
    p_parse.add_argument(
        "--output",
        default="outputs/parsed",
        help="Output root directory (default: outputs/parsed).",
    )
    p_parse.add_argument("--dpi", type=int, default=200, help="Render DPI (default: 200).")
    _add_ocr_flags(p_parse)
    p_parse.add_argument(
        "--list-models",
        action="store_true",
        help="List available backends and exit.",
    )
    p_parse.set_defaults(func=cmd_parse)

    p_eval = sub.add_parser(
        "evaluate",
        help="Score one model against a dataset; per-run results accumulate.",
    )
    p_eval.add_argument(
        "--track",
        required=True,
        choices=["pipeline", "vlm", "figure-link", "page-genre"],
        help="Which track to evaluate.",
    )
    p_eval.add_argument(
        "--dataset",
        required=True,
        choices=["glasbena_mladina", "doclaynet", "publaynet", "omnidocbench"],
        help="Dataset name.",
    )
    p_eval.add_argument(
        "--data-root",
        help="Override dataset root (e.g. default: datasets/glasbena_mladina for glasbena_mladina, "
             "datasets/doclaynet for doclaynet, datasets/OmniDocBench for omnidocbench).",
    )
    p_eval.add_argument(
        "--pdfs-dir",
        help="Directory of PDFs. Used by the VLM track to parse documents, "
             "and by the pipeline track to fill OmniDocBench `text` fields with "
             "real PDF text (falls back to stub tokens when no PDF resolves). "
             "Default: <data_root>/pdfs.",
    )
    p_eval.add_argument(
        "--images-dir",
        help="Directory of images for pipeline evaluation (default: <data_root>/images).",
    )
    p_eval.add_argument("--layout", help="Layout backend (pipeline track).")
    p_eval.add_argument("--order", default="top-bottom", help="Reading-order backend.")
    p_eval.add_argument("--vlm", help="VLM backend (vlm track).")
    p_eval.add_argument("--config", help="JSON config file for the chosen backend.")
    p_eval.add_argument(
        "--order-config",
        help="JSON config file for the reading-order backend (see `parse`).",
    )
    p_eval.add_argument(
        "--run-id",
        help="Run directory name under --output (default: current timestamp). "
             "Reuse the same run-id across invocations to accumulate models in one report.",
    )
    p_eval.add_argument(
        "--output",
        default="outputs/evaluations",
        help="Output root (default: outputs/evaluations).",
    )
    # Pipeline track: re-read the regions the text layer failed on before the
    # predicted markdown is scored. Predictions only — see `rare.evaluate.pdf_text`.
    _add_ocr_flags(p_eval)
    p_eval.add_argument("--limit", type=int, help="Cap number of samples (for smoke tests).")
    p_eval.add_argument("--start", type=int, help="Start index for evaluating samples.")
    p_eval.add_argument(
        "--emit-omnidocbench",
        dest="emit_omnidocbench",
        action="store_true",
        default=True,
        help="Also write OmniDocBench-shaped gt.json + per-model predictions JSON (default: on).",
    )
    p_eval.add_argument(
        "--no-emit-omnidocbench",
        dest="emit_omnidocbench",
        action="store_false",
        help="Disable the OmniDocBench export.",
    )
    p_eval.add_argument(
        "--category-map",
        help="Optional JSON file of {source_category_name: omnidocbench_category_type} "
             "merged on top of the built-in default map.",
    )
    p_eval.add_argument(
        "--run-omnidocbench",
        dest="run_omnidocbench",
        action="store_true",
        default=False,
        help="Evaluate using OmniDocBench.",
    )
    p_eval.add_argument(
        "--omnidocbench-ground",
        help="OmniDocBench ground truths file.",
    )
    p_eval.add_argument(
        "--run-manual",
        dest="run_omnidocbench",
        action="store_false",
        help="Manual evaluation.",
    )
    p_eval.add_argument(
        "--omnidocbench-image",
        help="Override the OmniDocBench Docker image used for the end2end pass "
             "(default: the pinned repro image).",
    )
    p_eval.add_argument(
        "--omnidocbench-layout-image",
        help="Override the OmniDocBench Docker image used for the detection pass "
             "(default: the pinned layout image). The two passes ship in different "
             "images, so they are configured separately.",
    )
    p_eval.add_argument(
        "--omnidocbench-eval",
        choices=["detection", "end2end", "both"],
        default=None,
        help="Pipeline-track: which OmniDocBench pass(es) --run-omnidocbench runs. "
             "'detection' scores boxes (COCODet mAP); 'end2end' scores the rendered "
             "per-page Markdown (text_block/reading_order Edit_dist), the same way the "
             "VLM track is scored, and needs a resolvable --pdfs-dir for real text. "
             "Default: both.",
    )
    p_eval.add_argument(
        "--list-models",
        action="store_true",
        help="List backends and exit (same as `rare parse --list-models`).",
    )

    # --- figure-link / page-genre tracks -----------------------------------
    p_eval.add_argument(
        "--annotations",
        help="figure-link/page-genre tracks: COCO annotations (figure-link needs "
             "`order_id`, page-genre needs `page_type`). Default: "
             "<data_root>/annotations_with_order.json, else annotations.json.",
    )
    p_eval.add_argument(
        "--docs-dir",
        help="figure-link/page-genre tracks: score the `*_doc.json` files under this "
             "directory. figure-link requires them and defaults to outputs/parsed/gt; "
             "page-genre scores them end to end, and omitting it there builds "
             "documents from the ground-truth layout and order instead.",
    )
    p_eval.add_argument(
        "--variants",
        help="figure-link track: comma-separated config variants to score side by "
             "side (default: full,geometry,ner,nearest — see "
             "rare.evaluate.figure_link.VARIANTS).",
    )
    p_eval.add_argument(
        "--page-type-map",
        help="page-genre track: JSON file of {page_type: genre | [genres] | null} "
             "merged over the built-in map; null leaves that page type out of the "
             "score (see rare.evaluate.page_genre.PAGE_TYPE_TO_GENRE).",
    )
    p_eval.add_argument(
        "--ner",
        help="figure-link/page-genre tracks: NER backend for the linking passes "
             "(default: none).",
    )
    p_eval.add_argument(
        "--classification",
        help="figure-link/page-genre tracks: article-genre backend for the linking "
             "passes (default: none, which leaves page-genre with the genres "
             "rare.link.classify reads off the running headers).",
    )
    p_eval.add_argument(
        "--link-config",
        help="JSON config file for the linking passes (see `parse --link-config`).",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    # `tools` is documented here so `rare --help` shows it, but argv is
    # peeled off in main() before argparse runs so the underlying helper
    # can own its own flag namespace.
    sub.add_parser(
        "tools",
        help="Annotation utilities (was helper.py). Try `rare tools -m count-annotations -a <file>` or `python -m rare.tools._helper -h` for the full flag list.",
        add_help=False,
    ).set_defaults(func=cmd_tools)

    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(argv) if argv is not None else sys.argv[1:]

    # Special-case: pass `rare tools <...>` directly to the helper's argparse
    # so we don't have to mirror its (~10) flags here.
    if argv and argv[0] == "tools":
        from rare.tools._helper import main as tools_main
        sys.exit(tools_main(argv[1:]) or 0)

    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    sys.exit(code or 0)


if __name__ == "__main__":
    main()
