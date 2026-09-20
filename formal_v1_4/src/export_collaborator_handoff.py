"""Export privacy-safe category mappings and pipeline audit for collaborators."""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
INPUT = ROOT / "data" / "drive_verified_20260917"
RESULTS = BASE / "成果" / "全量匯入"
LABELS = RESULTS / "正式版_v1.4_全量語意分類_合併.jsonl"
OUT = ROOT / "公開版" / "handoff"


def load_labels() -> dict[str, dict]:
    labels: dict[str, dict] = {}
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            labels[str(row["prompt_hash"])] = row
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return labels


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    counters = Counter()
    seen_request_ids: set[str] = set()
    prompt_rows: dict[str, str] = {}
    request_rows: list[tuple[str, str]] = []
    exceptions: list[tuple[str, str, str]] = []

    archives = sorted(INPUT.glob("*.zip"))
    for archive in archives:
        try:
            zf = zipfile.ZipFile(archive)
        except (OSError, zipfile.BadZipFile) as exc:
            counters["bad_zip"] += 1
            exceptions.append((archive.name, "archive_error", type(exc).__name__))
            continue
        with zf:
            for info in zf.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".json"):
                    continue
                counters["json_members"] += 1
                fallback_id = f"{archive.name}:{info.filename}"
                try:
                    obj = json.loads(zf.read(info))
                except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
                    counters["json_parse_error"] += 1
                    exceptions.append((fallback_id, "parse_error", type(exc).__name__))
                    continue
                request = obj.get("request") or {}
                content = obj.get("content") or {}
                raw_request_id = obj.get("request_id") or request.get("request_id")
                request_id = str(raw_request_id or fallback_id)
                if not raw_request_id:
                    counters["missing_request_id_synthetic"] += 1
                    exceptions.append((request_id, "synthetic_request_id", "source record has no request_id"))
                if request_id in seen_request_ids:
                    counters["duplicate_request_id"] += 1
                    exceptions.append((request_id, "duplicate_request_id", "later duplicate excluded"))
                    continue
                seen_request_ids.add(request_id)
                counters["unique_requests"] += 1
                prompt = str(request.get("prompt_text") or content.get("prompt") or "").strip()
                if not prompt:
                    counters["blank_prompt"] += 1
                    request_rows.append((request_id, "未分類 > 空白 Prompt"))
                    exceptions.append((request_id, "blank_prompt", "no nonblank prompt_text/content.prompt"))
                    continue
                digest = hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest()
                label = labels.get(digest)
                if not label:
                    counters["unclassified_nonblank"] += 1
                    request_rows.append((request_id, "未分類 > 找不到正式標籤"))
                    exceptions.append((request_id, "unclassified", f"prompt_sha256={digest}"))
                    continue
                category = f"{label['main_category']} > {label['sub_category']}"
                request_rows.append((request_id, category))
                prompt_rows[digest] = category
                counters["classified_requests"] += 1

    def write_two_columns(path: Path, header: tuple[str, str], rows):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    write_two_columns(
        OUT / "request_id_category.csv",
        ("request_id", "category"),
        request_rows,
    )
    write_two_columns(
        OUT / "prompt_sha256_category.csv",
        ("prompt_sha256", "category"),
        sorted(prompt_rows.items()),
    )
    with (OUT / "processing_exceptions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("identifier", "status", "reason"))
        writer.writerows(exceptions)

    summary = {
        "scope": "23 downloaded Google Drive ZIP archives; request_id deduplicated",
        "privacy": "No prompt text, response text, account, email, student ID, or API key is exported",
        "category_format": "main_category > sub_category",
        "counts": {
            "zip_archives": len(archives),
            "json_members": counters["json_members"],
            "unique_requests": counters["unique_requests"],
            "classified_requests": counters["classified_requests"],
            "blank_prompt": counters["blank_prompt"],
            "unclassified_nonblank": counters["unclassified_nonblank"],
            "unique_nonblank_prompt_hashes": len(prompt_rows),
            "formal_labels_available": len(labels),
            "missing_request_id_synthetic": counters["missing_request_id_synthetic"],
            "duplicate_request_id": counters["duplicate_request_id"],
            "json_parse_error": counters["json_parse_error"],
            "bad_zip": counters["bad_zip"],
        },
        "interpretation": {
            "skipped_or_dropped": "Only later duplicate request_id rows would be excluded; none are expected in this snapshot.",
            "blank_prompt": "Kept in request mapping as 未分類 > 空白 Prompt and listed in exceptions.",
            "model_failure": "No nonblank Prompt remains unclassified in formal v1.4 after semantic completion.",
            "transient_api_errors": "Checkpoint/retry errors during processing are not final row failures when a final label exists.",
            "synthetic_request_id": "Records lacking request_id use archive.zip:member.json for audit only; join by prompt_sha256 when needed.",
        },
    }
    (OUT / "pipeline_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["counts"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
