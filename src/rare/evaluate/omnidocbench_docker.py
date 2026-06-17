"""Auto-invoke OmniDocBench's end2end evaluator in its pinned Docker container.

This is the "approach C" wiring: `rare evaluate --track pipeline` already emits
the artifacts OmniDocBench needs (`gt.json` + a per-page markdown directory, see
`rare.evaluate.runner.run_pipeline`). This module runs the canonical container
against those artifacts and parses the resulting Edit-distance numbers back so
they land in the run's `report.md` — turning the previously-manual
`scripts/omnidocbench/run.sh` step into part of one `rare evaluate` command.

We deliberately reuse the *exact* container, mounts, and config heredoc from
`scripts/omnidocbench/run.sh` so the numbers match what that script produces.
The container is the reproducibility boundary (it carries the CDM/LaTeX stack);
nothing here imports OmniDocBench in-process. A future "approach A" would swap
this subprocess for a local install — keep that seam in mind.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

# The pinned image used by scripts/omnidocbench/run.sh. Kept in one place so
# both paths bump together.
DEFAULT_IMAGE = "ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204"
DEFAULT_LAYOUT_IMAGE = "omnidocbench-v15:latest"

# OmniDocBench names result files `<pred_folder>_<match_method>_*`. Since we
# mount our markdown dir at `.../data_md/predictions`, the prefix is fixed.
_RESULT_PREFIX = "predictions_quick_match"
_METRIC_RESULT_FILE = f"{_RESULT_PREFIX}_metric_result.json"


def _build_config_yaml() -> str:
    """The end2end config, mirroring run.sh minus the `display_formula` block.
    CDM (the formula visual metric) is the only part that needs the heavy
    in-container LaTeX stack and it's irrelevant for formula-free documents
    (e.g. magazines), so we score only `text_block` and `reading_order`."""
    return """end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: ./gt/your_gt.json
    prediction:
      data_path: ./data_md/predictions
    match_method: quick_match
    match_workers: 4
    quick_match_truncated_timeout_sec: 300
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
"""

def _build_config_layout_yaml(pred_cat_mapping: Optional[str] = """\
                              title : title
                              plain text: text
                              abandon: abandon
                              figure: figure
                              figure_caption: figure_caption
                              """) -> str:
    """The end2end config, mirroring run.sh minus the `display_formula` block.
    CDM (the formula visual metric) is the only part that needs the heavy
    in-container LaTeX stack and it's irrelevant for formula-free documents
    (e.g. magazines), so we score only `text_block` and `reading_order`."""
    return f"""detection_eval:   # Specify task name, common for all detection-related tasks
  metrics:
    - COCODet     # Detection task related metrics, mainly mAP, mAR etc.
  dataset:
    dataset_name: detection_dataset_simple_format       # Dataset name, no need to modify if following specified input format
    ground_truth:
      data_path: ./demo_data/omnidocbench_demo/OmniDocBench_demo.json               # Path to OmniDocBench JSON file
    prediction:
      data_path: ./demo_data/detection/detection_prediction.json                    # Path to model prediction result JSON file
  categories:
    eval_cat:                # Categories participating in final evaluation
      block_level:           # Block level categories, see OmniDocBench evaluation set introduction for details
        - title              # Title
        - text               # Text
        - abandon            # Includes headers, footers, page numbers, and page annotations
        - figure             # Image
        - figure_caption     # Image caption
    gt_cat_mapping:          # Mapping table from ground truth to final evaluation categories, key is ground truth category, value is final evaluation category name
      figure_footnote: figure_footnote
      figure_caption: figure_caption
      page_number: abandon
      header: abandon
      page_footnote: abandon
      refernece: text
      figure: figure
      title: title
      text_block: text
      footer: abandon
    pred_cat_mapping:       # Mapping table from prediction to final evaluation categories, key is prediction category, value is final evaluation category name
      f{pred_cat_mapping}
"""


def _docker_command(
    gt_path: Path,
    pred_md_dir: Path,
    result_dir: Path,
    image: str,
    type: str='end2end',
    omnidocbench_pred_cat_mapping: Optional[str] = None,
) -> list[str]:
    """Assemble the `docker run ... bash -c <heredoc>` invocation from run.sh."""
    if type == 'end2end':
        config_yaml = _build_config_yaml()
    else:
        config_yaml = _build_config_layout_yaml(omnidocbench_pred_cat_mapping)
    # Same shape as run.sh: write the config inside the container, then run the
    # validator. The heredoc keeps us from needing an extra host file + mount.
    inner = (
        'cat > configs/custom.yaml << "EOF"\n'
        f"{config_yaml}"
        "EOF\n"
        "python pdf_validation.py --config configs/custom.yaml"
    )
    return [
        "docker", "run", "--rm",
        "--entrypoint", "bash",
        "-v", f"{gt_path}:/workspace/gt/your_gt.json:ro",
        "-v", f"{pred_md_dir}:/workspace/data_md/predictions:ro",
        "-v", f"{result_dir}:/workspace/result",
        image,
        "-c", inner,
    ]


def _as_float(v) -> Optional[float]:
    """OmniDocBench writes "NaN" (string) when a metric has no samples; treat
    that and anything non-numeric as missing rather than a 0.0 score."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN


def parse_metric_result(result_dir: Path) -> dict[str, float]:
    """Read `<result_dir>/predictions_quick_match_metric_result.json` and pull
    out the headline Edit-distance numbers as a flat `{metric: value}` dict
    suitable for merging into a per-model `aggregates` block.
omnidocbench_pred_cat_mapping
    We surface the per-sample average (OmniDocBench's standard headline figure)
    for `text_block` and `reading_order`. Missing/NaN metrics are omitted.
    """
    path = result_dir / _METRIC_RESULT_FILE
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: dict[str, float] = {}
    for section, key in (
        ("text_block", "odb_text_block_edit"),
        ("reading_order", "odb_reading_order_edit"),
    ):
        edit = (
            data.get(section, {})
            .get("all", {})
            .get("Edit_dist", {})
        )
        val = _as_float(edit.get("edit_sample_avg"))
        if val is None:
            val = _as_float(edit.get("ALL_page_avg"))
        if val is not None:
            out[key] = val
    return out


def run_eval(
    gt_path: Path,
    pred_md_dir: Path,
    result_dir: Path,
    image: str = DEFAULT_IMAGE,
    timeout: Optional[int] = None,
    type: str='end2end',
    omnidocbench_pred_cat_mapping: str=None,
) -> dict[str, float]:
    """Run the OmniDocBench container and return parsed Edit-distance metrics.

    Returns an empty dict (and prints a warning) on any failure — Docker
    missing, container non-zero exit, or absent result file — so a missing
    OmniDocBench run never sinks the local pipeline metrics already computed.
    """
    if shutil.which("docker") is None:
        print("[omnidocbench] docker not found on PATH; skipping container eval.")
        return {}

    gt_path = gt_path.resolve()
    pred_md_dir = pred_md_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    result_dir = result_dir.resolve()

    cmd = _docker_command(gt_path, pred_md_dir, result_dir, image, type=type, omnidocbench_pred_cat_mapping=omnidocbench_pred_cat_mapping)
    print(f"[omnidocbench] running container {image} ...")
    try:
        proc = subprocess.run(cmd, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"[omnidocbench] container run failed: {exc}")
        return {}
    if proc.returncode != 0:
        print(f"[omnidocbench] container exited with code {proc.returncode}; "
              f"see output above. Skipping metric merge.")
        return {}

    metrics = parse_metric_result(result_dir)
    if not metrics:
        print(f"[omnidocbench] no parseable metrics in {result_dir}.")
    else:
        print(f"[omnidocbench] {metrics}")
    return metrics