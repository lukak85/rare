"""Unified command-line interface: `rare parse | evaluate | tools`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from rare.models.registry import ensure_layoutparser_backend, get, list_backends


def _read_config(path: str | None) -> dict | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text())


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
        return 0

    if not args.pdf:
        print(
            "error: missing PDF path. Usage:\n"
            "  astr parse <pdf> --layout <name> [--order <name>]\n"
            "  astr parse <pdf> --vlm <name>",
            file=sys.stderr,
        )
        return 2

    if args.vlm and args.layout:
        print("error: pass either --vlm or --layout, not both.", file=sys.stderr)
        return 2
    if not args.vlm and not args.layout:
        print("error: one of --layout or --vlm is required.", file=sys.stderr)
        return 2

    if args.vlm:
        # VLM track — produces a GlasanaDocument directly.
        from rare.parse.io import write_outputs

        vlm_cls = get("vlm", args.vlm)
        vlm = vlm_cls(config=_read_config(args.config))
        doc = vlm.parse_pdf(args.pdf)
        out = write_outputs(doc, args.output)
        print(f"Output written to: {out}")
        return 0

    # Pipeline track. Instantiating the layout backend first ensures
    # LAYOUTPARSER_BACKEND is set before any layoutparser import.
    layout_cls = get("layout", args.layout)
    layout = layout_cls(config=_read_config(args.config))

    order_cls = get("order", args.order)
    order = order_cls()

    # Imported only after the layout backend is constructed.
    from rare.parse.pipeline import parse_pdf

    out = parse_pdf(args.pdf, layout, order, output_dir=args.output, dpi=args.dpi)
    print(f"Output written to: {out}")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    # Stub — never reached. `tools` is dispatched in main() before argparse runs
    # so the helper can own its own flag namespace.
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    if args.list_models:
        return cmd_parse(args)  # reuse the same listing

    # Must run before dataset loading: gold-layout construction imports
    # layoutparser, which freezes LAYOUTPARSER_BACKEND for the process.
    if args.track == "pipeline" and args.layout:
        ensure_layoutparser_backend(args.layout)

    from rare.evaluate import datasets as ds_loader

    dataset = ds_loader.load(args.dataset, root=args.data_root) if args.data_root \
              else ds_loader.load(args.dataset)

    run_id = args.run_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.track == "pipeline":
        if not args.layout:
            print("error: --layout required for --track pipeline.", file=sys.stderr)
            return 2
        layout_cls = get("layout", args.layout)
        layout = layout_cls(config=_read_config(args.config))
        order_cls = get("order", args.order)
        order = order_cls()

        from rare.evaluate.runner import run_pipeline
        agg = run_pipeline(dataset, layout, order, run_dir, limit=args.limit)

    elif args.track == "vlm":
        if not args.vlm:
            print("error: --vlm required for --track vlm.", file=sys.stderr)
            return 2
        vlm_cls = get("vlm", args.vlm)
        vlm = vlm_cls(config=_read_config(args.config))

        from rare.evaluate.runner import run_vlm
        pdfs_dir = Path(args.pdfs_dir) if args.pdfs_dir else None
        agg = run_vlm(dataset, vlm, run_dir, pdfs_dir=pdfs_dir, limit=args.limit)

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
        "--config",
        help="JSON config file passed to the chosen backend.",
    )
    p_parse.add_argument(
        "--output",
        default="outputs/parsed",
        help="Output root directory (default: outputs/parsed).",
    )
    p_parse.add_argument("--dpi", type=int, default=200, help="Render DPI (default: 200).")
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
        choices=["pipeline", "vlm"],
        help="Which track to evaluate.",
    )
    p_eval.add_argument(
        "--dataset",
        required=True,
        choices=["glasbena_mladina", "doclaynet", "publaynet"],
        help="Dataset name.",
    )
    p_eval.add_argument(
        "--data-root",
        help="Override dataset root (e.g. default: datasets/glasbena_mladina for glasbena_mladina, "
             "datasets/doclaynet for doclaynet).",
    )
    p_eval.add_argument(
        "--pdfs-dir",
        help="Directory of PDFs for VLM evaluation (default: <data_root>/pdfs).",
    )
    p_eval.add_argument("--layout", help="Layout backend (pipeline track).")
    p_eval.add_argument("--order", default="top-bottom", help="Reading-order backend.")
    p_eval.add_argument("--vlm", help="VLM backend (vlm track).")
    p_eval.add_argument("--config", help="JSON config file for the chosen backend.")
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
    p_eval.add_argument("--limit", type=int, help="Cap number of samples (for smoke tests).")
    p_eval.add_argument(
        "--list-models",
        action="store_true",
        help="List backends and exit (same as `rare parse --list-models`).",
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
