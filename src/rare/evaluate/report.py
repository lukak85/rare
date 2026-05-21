"""Write evaluation results as Markdown + CSV."""

from __future__ import annotations

import csv
from pathlib import Path


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_report(
    output_dir: Path,
    track: str,
    dataset_name: str,
    aggregates: dict[str, dict[str, float]],
    per_image_rows: list[dict],
) -> None:
    """Write report.md and scores.csv into `output_dir`.

    aggregates:    {model_name: {metric: value}}
    per_image_rows: rows for scores.csv; each row a dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- report.md ------------------------------------------------------
    metric_keys: list[str] = []
    seen: set[str] = set()
    for scores in aggregates.values():
        for k in scores:
            if k not in seen:
                seen.add(k)
                metric_keys.append(k)

    lines = [
        f"# `rare evaluate --track {track}` — {dataset_name}",
        "",
        "| Model | " + " | ".join(metric_keys) + " |",
        "|" + "---|" * (len(metric_keys) + 1),
    ]
    for model, scores in aggregates.items():
        row = [model] + [_fmt(scores.get(k, "—")) for k in metric_keys]
        lines.append("| " + " | ".join(row) + " |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")

    # ----- scores.csv -----------------------------------------------------
    if per_image_rows:
        keys: list[str] = []
        seen.clear()
        for row in per_image_rows:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        with open(output_dir / "scores.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in per_image_rows:
                writer.writerow(row)
