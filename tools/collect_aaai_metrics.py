#!/usr/bin/env python3

import argparse
import glob
import json
import math
import os
import re
import sys


CORE_METRICS = {
    "mAP": "img_bbox_NuScenes/mAP",
    "NDS": "img_bbox_NuScenes/NDS",
    "AMOTA": "img_bbox_NuScenes/amota",
    "mAP_normal": "mAP_normal",
    "car_EPA": "car_EPA",
    "pedestrian_EPA": "pedestrian_EPA",
    "obj_box_col": "obj_box_col",
    "L2": "L2",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect standard/masked metrics for the AAAI masked-robustness protocol."
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        help="Experiment directories that contain evals/<checkpoint>/{standard,masked}/e2e_metrics.json.",
    )
    parser.add_argument(
        "--baseline-standard",
        help="Optional e2e_metrics.json path for the standard-eval baseline.",
    )
    parser.add_argument(
        "--baseline-masked",
        help="Optional e2e_metrics.json path for the masked-eval baseline.",
    )
    parser.add_argument(
        "--json-out",
        help="Optional output path for the collected summary JSON.",
    )
    return parser.parse_args()


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = payload.get("summary_metrics", payload)
    metrics = {}
    for short_name, key in CORE_METRICS.items():
        metrics[short_name] = summary.get(key)
    return metrics


def metric_number(value, default):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Unsupported metric value: {value!r}")


def iter_sort_key(name):
    match = re.search(r"(\d+)", name)
    if match:
        return int(match.group(1))
    return sys.maxsize


def collect_experiment(experiment_dir):
    pattern = os.path.join(experiment_dir, "evals", "*", "*", "e2e_metrics.json")
    records = {}
    for metrics_path in glob.glob(pattern):
        mode = os.path.basename(os.path.dirname(metrics_path))
        if mode not in {"standard", "masked"}:
            continue
        checkpoint = os.path.basename(os.path.dirname(os.path.dirname(metrics_path)))
        checkpoint_record = records.setdefault(checkpoint, {})
        checkpoint_record[mode] = load_metrics(metrics_path)
        checkpoint_record.setdefault("paths", {})[mode] = metrics_path
    return records


def choose_best_masked_l2(records):
    candidates = [
        (checkpoint, record["masked"])
        for checkpoint, record in records.items()
        if "masked" in record
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            metric_number(item[1].get("L2"), math.inf),
            metric_number(item[1].get("obj_box_col"), math.inf),
            -metric_number(item[1].get("NDS"), -math.inf),
            -metric_number(item[1].get("AMOTA"), -math.inf),
        ),
    )[0]


def choose_best_masked_safe(records):
    candidates = [
        (checkpoint, record["masked"])
        for checkpoint, record in records.items()
        if "masked" in record
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            metric_number(item[1].get("obj_box_col"), math.inf),
            metric_number(item[1].get("L2"), math.inf),
            -metric_number(item[1].get("NDS"), -math.inf),
        ),
    )[0]


def choose_best_standard(records):
    candidates = [
        (checkpoint, record["standard"])
        for checkpoint, record in records.items()
        if "standard" in record
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            -metric_number(item[1].get("NDS"), -math.inf),
            metric_number(item[1].get("L2"), math.inf),
            -metric_number(item[1].get("AMOTA"), -math.inf),
            metric_number(item[1].get("obj_box_col"), math.inf),
        ),
    )[0]


def delta(candidate_value, baseline_value):
    if candidate_value is None or baseline_value is None:
        return None
    return candidate_value - baseline_value


def format_metric(value, percentage=False):
    if value is None:
        return "-"
    if percentage:
        return f"{value * 100:.3f}%"
    return f"{value:.4f}"


def emit_markdown(name, records, bests, baseline_standard, baseline_masked):
    lines = []
    lines.append(f"## {name}")
    lines.append("")
    lines.append("| checkpoint | std NDS | std L2 | std AMOTA | masked NDS | masked L2 | masked obj_box_col |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for checkpoint in sorted(records.keys(), key=iter_sort_key):
        record = records[checkpoint]
        std_metrics = record.get("standard", {})
        masked_metrics = record.get("masked", {})
        lines.append(
            "| {checkpoint} | {std_nds} | {std_l2} | {std_amota} | {masked_nds} | {masked_l2} | {masked_obj_col} |".format(
                checkpoint=checkpoint,
                std_nds=format_metric(std_metrics.get("NDS")),
                std_l2=format_metric(std_metrics.get("L2")),
                std_amota=format_metric(std_metrics.get("AMOTA")),
                masked_nds=format_metric(masked_metrics.get("NDS")),
                masked_l2=format_metric(masked_metrics.get("L2")),
                masked_obj_col=format_metric(masked_metrics.get("obj_box_col"), percentage=True),
            )
        )
    lines.append("")
    lines.append(f"- best_masked_l2: `{bests['best_masked_l2']}`")
    lines.append(f"- best_masked_safe: `{bests['best_masked_safe']}`")
    lines.append(f"- best_standard: `{bests['best_standard']}`")

    if baseline_standard is not None and bests["best_standard"] is not None:
        best_standard_metrics = records[bests["best_standard"]]["standard"]
        lines.append(
            "- standard delta vs baseline: NDS {nds}, L2 {l2}".format(
                nds=format_metric(delta(best_standard_metrics.get("NDS"), baseline_standard.get("NDS"))),
                l2=format_metric(delta(best_standard_metrics.get("L2"), baseline_standard.get("L2"))),
            )
        )
    if baseline_masked is not None and bests["best_masked_l2"] is not None:
        best_masked_metrics = records[bests["best_masked_l2"]]["masked"]
        lines.append(
            "- masked delta vs baseline: NDS {nds}, L2 {l2}, obj_box_col {obj_box_col}".format(
                nds=format_metric(delta(best_masked_metrics.get("NDS"), baseline_masked.get("NDS"))),
                l2=format_metric(delta(best_masked_metrics.get("L2"), baseline_masked.get("L2"))),
                obj_box_col=format_metric(
                    delta(best_masked_metrics.get("obj_box_col"), baseline_masked.get("obj_box_col")),
                    percentage=True,
                ),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    baseline_standard = load_metrics(args.baseline_standard) if args.baseline_standard else None
    baseline_masked = load_metrics(args.baseline_masked) if args.baseline_masked else None

    all_results = {}
    markdown_chunks = []
    for experiment_dir in args.experiment_dirs:
        records = collect_experiment(experiment_dir)
        bests = {
            "best_masked_l2": choose_best_masked_l2(records),
            "best_masked_safe": choose_best_masked_safe(records),
            "best_standard": choose_best_standard(records),
        }
        all_results[experiment_dir] = {
            "records": records,
            "bests": bests,
        }
        markdown_chunks.append(
            emit_markdown(
                experiment_dir,
                records,
                bests,
                baseline_standard=baseline_standard,
                baseline_masked=baseline_masked,
            )
        )

    markdown_output = "\n".join(markdown_chunks).strip()
    if markdown_output:
        print(markdown_output)

    if args.json_out:
        payload = {
            "baseline_standard": baseline_standard,
            "baseline_masked": baseline_masked,
            "experiments": all_results,
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
