from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / "成果" / "prompt_classifications.jsonl"
DEFAULT_USAGE_INPUT = BASE_DIR / "成果" / "classification_api_usage.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "成果" / "彙總"
TOKEN_FIELDS = ["prompt_tokens", "cached_tokens", "cache_write_tokens", "completion_tokens", "total_tokens"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as file_handle:
        for line in file_handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def add_metrics(target: dict[str, Any], record: dict[str, Any], allocation: float) -> None:
    target["prompt_count"] += 1
    if record.get("account_hash"):
        target["users"].add(str(record["account_hash"]))
    for field in TOKEN_FIELDS:
        value = number(record.get(field))
        target[f"duplicated_{field}"] += value
        target[f"allocated_{field}"] += value * allocation
    if record.get("estimated_cost_usd") is not None:
        cost = number(record.get("estimated_cost_usd"))
        target["duplicated_cost_usd"] += cost
        target["allocated_cost_usd"] += cost * allocation


def metric_row() -> dict[str, Any]:
    row: dict[str, Any] = {
        "prompt_count": 0,
        "users": set(),
        "duplicated_cost_usd": 0.0,
        "allocated_cost_usd": 0.0,
    }
    for field in TOKEN_FIELDS:
        row[f"duplicated_{field}"] = 0.0
        row[f"allocated_{field}"] = 0.0
    return row


def flatten_metrics(row: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in row.items() if key != "users"}
    result["unique_users"] = len(row["users"])
    for key, value in list(result.items()):
        if key.startswith("allocated_") or key.endswith("_cost_usd"):
            result[key] = round(float(value), 6)
        elif key.startswith("duplicated_"):
            result[key] = int(round(float(value)))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def label_entries(record: dict[str, Any], axis: str) -> list[tuple[str, str]]:
    classification = record.get("classification") or {}
    if axis == "technologies":
        return [("技術", str(value)) for value in classification.get("technologies") or []]
    return [
        (str(item.get("major") or ""), str(item.get("minor") or ""))
        for item in classification.get(axis) or []
    ]


def aggregate_axis(records: list[dict[str, Any]], axis: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(metric_row)
    for record in records:
        labels = list(dict.fromkeys(label_entries(record, axis)))
        if not labels:
            continue
        allocation = 1.0 / len(labels)
        for key in labels:
            add_metrics(grouped[key], record, allocation)
    rows = []
    for (major, minor), metrics in grouped.items():
        rows.append({"major": major, "minor": minor, **flatten_metrics(metrics)})
    return sorted(rows, key=lambda row: (-row["duplicated_total_tokens"], row["major"], row["minor"]))


def aggregate_domain_task(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(metric_row)
    for record in records:
        domains = list(dict.fromkeys(label_entries(record, "domains")))
        tasks = list(dict.fromkeys(label_entries(record, "tasks")))
        combinations = [(domain, task) for domain in domains for task in tasks]
        if not combinations:
            continue
        allocation = 1.0 / len(combinations)
        for domain, task in combinations:
            key = (domain[0], domain[1], task[0], task[1])
            add_metrics(grouped[key], record, allocation)
    rows = []
    for key, metrics in grouped.items():
        rows.append({
            "domain_major": key[0], "domain_minor": key[1],
            "task_major": key[2], "task_minor": key[3],
            **flatten_metrics(metrics),
        })
    return sorted(rows, key=lambda row: (-row["duplicated_total_tokens"], row["domain_major"], row["task_major"]))


def api_usage_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    return {
        "api_calls": len(rows),
        "classified_unique_prompts": sum(
            int(row.get("unique_prompt_count") or row.get("prompt_count") or 0) for row in rows
        ),
        "expanded_records": sum(
            int(row.get("expanded_record_count") or row.get("prompt_count") or 0) for row in rows
        ),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
        "cached_input_tokens": sum(int(row.get("cached_input_tokens") or 0) for row in rows),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="彙總階層式分類結果與原始 Codex Token。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--usage-input", type=Path, default=DEFAULT_USAGE_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    if not records:
        raise SystemExit(f"沒有分類結果：{args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for axis, filename in [
        ("domains", "領域彙總.csv"),
        ("tasks", "任務彙總.csv"),
        ("artifacts", "產出物彙總.csv"),
        ("technologies", "技術彙總.csv"),
    ]:
        write_csv(args.output_dir / filename, aggregate_axis(records, axis))
    write_csv(args.output_dir / "領域任務交叉彙總.csv", aggregate_domain_task(records))

    confidence = Counter()
    for record in records:
        value = number((record.get("classification") or {}).get("confidence"))
        confidence["高（>=0.80）" if value >= 0.8 else "中（0.60–0.79）" if value >= 0.6 else "低（<0.60）"] += 1
    summary = {
        "classified_prompts": len(records),
        "unique_prompt_hashes": len({str(record.get("prompt_hash") or record.get("id")) for record in records}),
        "internal_prompts": sum(bool((record.get("classification") or {}).get("is_internal_prompt")) for record in records),
        "needs_review": sum(bool((record.get("classification") or {}).get("needs_review")) for record in records),
        "confidence_distribution": dict(confidence),
        "source_total_tokens": sum(int(record.get("total_tokens") or 0) for record in records),
        "classification_api_usage": api_usage_summary(args.usage_input),
        "counting_note": "duplicated 欄位允許同一 Prompt 在多分類重複計算；allocated 欄位在同一分類軸內平均分攤。",
        "quality_note": "未使用獨立人工金標時，不能將模型信心或模型間一致率稱為正確率。",
    }
    (args.output_dir / "分類總覽.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已建立彙總：{args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

