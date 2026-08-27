"""Recompute public mixed-reference metrics from the de-identified cell table."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def metrics(rows: list[dict[str, str]]) -> dict[str, float | int | None]:
    tp = fp = fn = tn = 0
    for row in rows:
        actual = row["mixed_reference"] == "是"
        predicted = row["regex_prediction"] == "是"
        if actual and predicted:
            tp += 1
        elif not actual and predicted:
            fp += 1
        elif actual and not predicted:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    accuracy = (tp + tn) / n if n else None
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def cluster_bootstrap(rows: list[dict[str, str]], repetitions: int, seed: int) -> dict[str, list[float]]:
    by_prompt: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_prompt[row["sample_id"]].append(row)
    prompt_ids = list(by_prompt)
    rng = random.Random(seed)
    accuracy_values: list[float] = []
    f1_values: list[float] = []
    for _ in range(repetitions):
        sample: list[dict[str, str]] = []
        for _ in prompt_ids:
            sample.extend(by_prompt[rng.choice(prompt_ids)])
        result = metrics(sample)
        accuracy_values.append(float(result["accuracy"] or 0))
        f1_values.append(float(result["f1"] or 0))
    return {
        "accuracy_95_ci": [percentile(accuracy_values, 0.025), percentile(accuracy_values, 0.975)],
        "f1_95_ci": [percentile(f1_values, 0.025), percentile(f1_values, 0.975)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["included_in_evaluation"] == "是" and row["mixed_reference"] in {"是", "否"} and row["regex_prediction"] in {"是", "否"}]
    result = metrics(rows)
    result.update(cluster_bootstrap(rows, args.bootstrap, args.seed))
    result["prompt_count"] = len({row["sample_id"] for row in rows})
    result["metric_name"] = "mixed_reference_agreement"
    result["interpretation"] = "Agreement with human plus dual-AI consensus labels; not independent human-test accuracy."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

