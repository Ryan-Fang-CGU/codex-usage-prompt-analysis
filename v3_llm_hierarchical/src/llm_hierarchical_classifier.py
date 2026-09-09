from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


VERSION = "3.0.0"
BASE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = BASE_DIR.parent
DEFAULT_TAXONOMY = BASE_DIR / "config" / "taxonomy.json"
DEFAULT_OUTPUT = BASE_DIR / "成果" / "prompt_classifications.jsonl"
DEFAULT_USAGE_OUTPUT = BASE_DIR / "成果" / "classification_api_usage.jsonl"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
API_URL = "https://api.openai.com/v1/responses"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def stable_hash(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:length]


def make_record(
    obj: dict[str, Any],
    source_ref: str,
    *,
    zipped: bool,
) -> dict[str, Any] | None:
    identity = obj.get("identity") or {}
    usage = obj.get("usage") or {}

    if zipped:
        request = obj.get("request") or {}
        response = obj.get("response") or {}
        prompt = str(request.get("prompt_text") or "")
        created_at = obj.get("created_at") or obj.get("received_at")
        request_id = obj.get("request_id") or request.get("request_id")
        unit_id = request.get("thread_id") or request.get("conversation_id")
        model = response.get("model_returned") or request.get("model_requested")
        cached_tokens = safe_int(usage.get("cached_tokens"))
        cache_write_tokens = safe_int(usage.get("cache_write_tokens"))
    else:
        request = obj.get("request") or {}
        model_info = obj.get("model") or {}
        conversation = obj.get("conversation") or {}
        details = obj.get("usage_details") or {}
        input_details = details.get("input_tokens_details") or {}
        prompt = str((obj.get("content") or {}).get("prompt") or "")
        created_at = (obj.get("time") or {}).get("created_at") or (obj.get("time") or {}).get("received_at")
        request_id = request.get("request_id")
        unit_id = conversation.get("conversation_key") or conversation.get("thread_id")
        model = model_info.get("model_returned") or model_info.get("model_requested")
        cached_tokens = safe_int(details.get("cached_tokens") or input_details.get("cached_tokens"))
        cache_write_tokens = safe_int(details.get("cache_write_tokens") or input_details.get("cache_write_tokens"))

    if not prompt.strip():
        return None

    user_identity = identity.get("user_account") or identity.get("anonymous_user_id") or "missing"
    prompt_hash = stable_hash(prompt, 64)
    record_id = stable_hash("|".join([source_ref, str(request_id or ""), prompt_hash]))
    return {
        "id": record_id,
        "request_id": str(request_id or record_id),
        "unit_id": str(unit_id or ""),
        "source_ref": source_ref,
        "created_at": created_at,
        "account_hash": stable_hash(str(user_identity), 14),
        "source_model": str(model or "missing"),
        "prompt": prompt,
        "prompt_hash": prompt_hash,
        "prompt_tokens": safe_int(usage.get("prompt_tokens")),
        "completion_tokens": safe_int(usage.get("completion_tokens")),
        "total_tokens": safe_int(usage.get("total_tokens")),
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
    }


def iter_data_records(data_dir: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(data_dir.rglob("*.clean.json")):
        try:
            obj = read_json(path)
            source_ref = path.relative_to(data_dir).as_posix()
            record = make_record(obj, source_ref, zipped=False)
            if record:
                yield record
        except (OSError, json.JSONDecodeError) as exc:
            print(f"略過無法讀取的檔案：{path.name}（{exc}）", file=sys.stderr)

    for zip_path in sorted(data_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path) as archive:
                for info in archive.infolist():
                    if info.is_dir() or not info.filename.lower().endswith(".json"):
                        continue
                    try:
                        with archive.open(info) as file_handle:
                            obj = json.load(file_handle)
                        source_ref = f"{zip_path.name}!{info.filename}"
                        record = make_record(obj, source_ref, zipped=True)
                        if record:
                            yield record
                    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                        print(f"略過壓縮檔項目：{zip_path.name}!{info.filename}（{exc}）", file=sys.stderr)
        except (OSError, zipfile.BadZipFile) as exc:
            print(f"略過無法讀取的壓縮檔：{zip_path.name}（{exc}）", file=sys.stderr)


def iter_input_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file_handle:
        for line_number, line in enumerate(file_handle, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = str(obj.get("prompt") or "")
            if not prompt.strip():
                continue
            prompt_hash = stable_hash(prompt, 64)
            record_id = str(obj.get("id") or stable_hash(f"{line_number}|{prompt_hash}"))
            metadata = {key: value for key, value in obj.items() if key not in {"id", "prompt"}}
            yield {"id": record_id, "prompt": prompt, "prompt_hash": prompt_hash, **metadata}


def truncate_prompt(prompt: str, max_chars: int) -> tuple[str, bool]:
    if len(prompt) <= max_chars:
        return prompt, False
    head = max_chars * 2 // 3
    tail = max_chars - head
    return prompt[:head] + "\n\n[中間內容因長度限制省略]\n\n" + prompt[-tail:], True


def output_schema(taxonomy: dict[str, Any]) -> dict[str, Any]:
    domain_majors = list(taxonomy["domains"])
    task_majors = list(taxonomy["tasks"])
    artifact_majors = list(taxonomy["artifacts"])

    label_object = lambda majors: {
        "type": "object",
        "properties": {
            "major": {"type": "string", "enum": majors},
            "minor": {"type": "string"},
            "detail": {"type": "string"},
        },
        "required": ["major", "minor", "detail"],
        "additionalProperties": False,
    }

    item = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "is_internal_prompt": {"type": "boolean"},
            "domains": {"type": "array", "items": label_object(domain_majors), "maxItems": 5},
            "tasks": {"type": "array", "items": label_object(task_majors), "maxItems": 7},
            "artifacts": {"type": "array", "items": label_object(artifact_majors), "maxItems": 7},
            "technologies": {
                "type": "array",
                "items": {"type": "string", "enum": taxonomy["technologies"]},
                "maxItems": 12,
            },
            "risk_flags": {
                "type": "array",
                "items": {"type": "string", "enum": taxonomy["risk_flags"]},
                "maxItems": 4,
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "needs_review": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": [
            "id", "is_internal_prompt", "domains", "tasks", "artifacts", "technologies",
            "risk_flags", "confidence", "needs_review", "reason"
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"items": {"type": "array", "items": item}},
        "required": ["items"],
        "additionalProperties": False,
    }


def classification_instructions(taxonomy: dict[str, Any]) -> str:
    compact = json.dumps(taxonomy, ensure_ascii=False, separators=(",", ":"))
    return (
        "你是大學 Codex 使用紀錄的階層式多標籤分類器。"
        "只做語意判斷，不使用字面關鍵字作為唯一依據。"
        "每個輸入 id 必須輸出且只能輸出一次，順序與輸入相同。"
        "major 與 minor 必須完全使用分類表中的既有名稱；detail 可用簡短自由文字補充。"
        "若同一 Prompt 跨領域或包含多個任務，全部列出，不可只選一個。"
        "產出物只在 Prompt 明示要求產生或編輯時標記。"
        "technologies 只標記明示或可直接確定的技術。"
        "若是 Codex 內部提示，is_internal_prompt=true，domains/tasks/artifacts/technologies 均輸出空陣列。"
        "confidence 是整體分類把握度；資訊不足、minor 無法穩定判定或內容嚴重依賴前文時 needs_review=true。"
        "risk_flags 若沒有風險只輸出[\"無\"]，有其他旗標時不要同時輸出\"無\"。"
        "reason 限一個簡短句子，不要重述 Prompt。\n\n"
        f"分類表：{compact}"
    )


def normalize_label_list(labels: list[dict[str, Any]], allowed: dict[str, list[str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for label in labels:
        major = str(label.get("major") or "")
        minor = str(label.get("minor") or "")
        detail = str(label.get("detail") or "").strip()
        if major not in allowed or minor not in allowed[major]:
            raise ValueError(f"分類不在 taxonomy：{major} > {minor}")
        key = (major, minor, detail)
        if key not in seen:
            seen.add(key)
            result.append({"major": major, "minor": minor, "detail": detail})
    return result


def validate_batch_result(
    parsed: dict[str, Any],
    input_ids: list[str],
    taxonomy: dict[str, Any],
) -> list[dict[str, Any]]:
    items = parsed.get("items")
    if not isinstance(items, list) or [str(item.get("id")) for item in items] != input_ids:
        raise ValueError("模型輸出的 id、數量或順序與輸入不一致")

    valid_technologies = set(taxonomy["technologies"])
    valid_risks = set(taxonomy["risk_flags"])
    for item in items:
        internal = bool(item["is_internal_prompt"])
        item["domains"] = normalize_label_list(item["domains"], taxonomy["domains"])
        item["tasks"] = normalize_label_list(item["tasks"], taxonomy["tasks"])
        item["artifacts"] = normalize_label_list(item["artifacts"], taxonomy["artifacts"])
        item["technologies"] = list(dict.fromkeys(str(value) for value in item["technologies"]))
        item["risk_flags"] = list(dict.fromkeys(str(value) for value in item["risk_flags"])) or ["無"]
        if not set(item["technologies"]) <= valid_technologies:
            raise ValueError("模型輸出 taxonomy 以外的技術")
        if not set(item["risk_flags"]) <= valid_risks:
            raise ValueError("模型輸出 taxonomy 以外的風險旗標")
        if "無" in item["risk_flags"] and len(item["risk_flags"]) > 1:
            raise ValueError("風險旗標不可同時包含『無』與其他項目")
        if internal and any([item["domains"], item["tasks"], item["artifacts"], item["technologies"]]):
            raise ValueError("Codex 內部提示不可歸入其他領域、任務、產出物或技術")
        if not internal and (not item["domains"] or not item["tasks"]):
            raise ValueError("一般 Prompt 至少需要一個領域與一個任務；資訊不足時請使用無法判定類別")
    return items


def extract_output_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in response.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text" and content.get("text"):
                texts.append(str(content["text"]))
    if not texts:
        raise ValueError("API 回應沒有 output_text")
    return "".join(texts)


def post_json(payload: dict[str, Any], api_key: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def classify_batch(
    batch: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    *,
    model: str,
    api_key: str,
    max_prompt_chars: int,
    timeout: int,
    retries: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt_items = []
    truncated: dict[str, bool] = {}
    for record in batch:
        text, was_truncated = truncate_prompt(str(record["prompt"]), max_prompt_chars)
        truncated[str(record["id"])] = was_truncated
        prompt_items.append({"id": str(record["id"]), "prompt": text})

    payload = {
        "model": model,
        "store": False,
        "instructions": classification_instructions(taxonomy),
        "input": json.dumps({"prompts": prompt_items}, ensure_ascii=False),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "prompt_classification_batch",
                "strict": True,
                "schema": output_schema(taxonomy),
            }
        },
        "max_output_tokens": max(2500, 900 * len(batch)),
    }

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = post_json(payload, api_key, timeout)
            parsed = json.loads(extract_output_text(response))
            items = validate_batch_result(parsed, [str(record["id"]) for record in batch], taxonomy)
            for item in items:
                item["prompt_truncated"] = truncated[item["id"]]
            return items, response
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            delay = min(60.0, (2 ** attempt) + random.random())
            print(f"批次失敗，{delay:.1f} 秒後重試：{exc}", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError(f"批次分類失敗：{last_error}")


def read_existing_results(path: Path) -> tuple[set[str], dict[str, dict[str, Any]]]:
    completed: set[str] = set()
    by_prompt_hash: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed, by_prompt_hash
    with path.open("r", encoding="utf-8-sig") as file_handle:
        for line in file_handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                completed.add(str(row["id"]))
                prompt_hash = str(row.get("prompt_hash") or "")
                classification = row.get("classification")
                if prompt_hash and isinstance(classification, dict):
                    by_prompt_hash[prompt_hash] = {
                        "taxonomy_version": row.get("taxonomy_version"),
                        "classifier_model": row.get("classifier_model"),
                        "classification_batch_id": row.get("classification_batch_id"),
                        "classification": classification,
                    }
            except (json.JSONDecodeError, KeyError):
                continue
    return completed, by_prompt_hash


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: str = "a") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8", newline="\n") as file_handle:
        for row in rows:
            file_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        file_handle.flush()


def batches(records: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for record in records:
        batch.append(record)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def response_usage_row(
    response: dict[str, Any], batch_id: str, model: str, prompt_count: int, expanded_record_count: int
) -> dict[str, Any]:
    usage = response.get("usage") or {}
    input_details = usage.get("input_tokens_details") or {}
    return {
        "batch_id": batch_id,
        "response_id": response.get("id"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": response.get("model") or model,
        "unique_prompt_count": prompt_count,
        "expanded_record_count": expanded_record_count,
        "input_tokens": safe_int(usage.get("input_tokens")),
        "cached_input_tokens": safe_int(input_details.get("cached_tokens")),
        "output_tokens": safe_int(usage.get("output_tokens")),
        "total_tokens": safe_int(usage.get("total_tokens")),
    }


def output_row(
    record: dict[str, Any],
    classification: dict[str, Any],
    *,
    taxonomy_version: str,
    classifier_model: str,
    batch_id: str,
    include_prompt: bool,
    prompt: str | None = None,
) -> dict[str, Any]:
    row = {key: value for key, value in record.items() if key != "prompt"}
    if include_prompt and prompt is not None:
        row["prompt"] = prompt
    item = copy.deepcopy(classification)
    item["id"] = str(record["id"])
    row.update({
        "taxonomy_version": taxonomy_version,
        "classifier_model": classifier_model,
        "classification_batch_id": batch_id,
        "classification": item,
    })
    return row


def build_dry_run_preview(
    records: list[dict[str, Any]], taxonomy: dict[str, Any], model: str, max_prompt_chars: int
) -> dict[str, Any]:
    prompts = []
    for record in records:
        prompt, truncated = truncate_prompt(str(record["prompt"]), max_prompt_chars)
        prompts.append({"id": record["id"], "prompt": prompt, "truncated": truncated})
    return {
        "model": model,
        "store": False,
        "instructions": classification_instructions(taxonomy),
        "input": {"prompts": prompts},
        "output_schema": output_schema(taxonomy),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 LLM 進行非正則、階層式、多標籤 Prompt 分類。")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data-dir", type=Path, default=WORKSPACE_DIR / "data", help="原始 Codex JSON/ZIP 資料夾")
    source.add_argument("--input-jsonl", type=Path, help="自訂 JSONL；每列至少包含 id 與 prompt")
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--usage-output", type=Path, default=DEFAULT_USAGE_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-prompt-chars", type=int, default=12000)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.0, help="每批之間等待秒數")
    parser.add_argument("--restart", action="store_true", help="清空舊輸出並重新開始")
    parser.add_argument("--include-prompt", action="store_true", help="在結果中保留 Prompt 原文（預設不保留）")
    parser.add_argument("--dry-run", action="store_true", help="只產生請求預覽，不呼叫 API")
    parser.add_argument("--dry-run-output", type=Path, default=BASE_DIR / "成果" / "request_preview.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.batch_size > 25:
        raise SystemExit("--batch-size 必須介於 1 到 25")
    taxonomy = read_json(args.taxonomy)
    if taxonomy.get("version") != VERSION:
        print(f"提醒：程式版本 {VERSION}，taxonomy 版本 {taxonomy.get('version')}", file=sys.stderr)

    records_source = iter_input_jsonl(args.input_jsonl) if args.input_jsonl else iter_data_records(args.data_dir)
    completed, classification_cache = (set(), {}) if args.restart else read_existing_results(args.output)
    groups: dict[str, dict[str, Any]] = {}
    reused_rows: list[dict[str, Any]] = []
    reused_count = 0
    for record in records_source:
        if str(record["id"]) in completed:
            continue
        prompt_hash = str(record["prompt_hash"])
        cached = classification_cache.get(prompt_hash)
        if cached:
            reused_rows.append(output_row(
                record,
                cached["classification"],
                taxonomy_version=str(cached.get("taxonomy_version") or taxonomy["version"]),
                classifier_model=str(cached.get("classifier_model") or args.model),
                batch_id=str(cached.get("classification_batch_id") or "prompt-hash-cache"),
                include_prompt=args.include_prompt,
                prompt=str(record["prompt"]),
            ))
            reused_count += 1
            if len(reused_rows) >= 5000 and not args.dry_run:
                write_jsonl(args.output, reused_rows)
                reused_rows = []
            continue
        if prompt_hash not in groups:
            if args.limit is not None and len(groups) >= args.limit:
                break
            groups[prompt_hash] = {"representative": record, "records": []}
        metadata = {key: value for key, value in record.items() if key != "prompt"}
        groups[prompt_hash]["records"].append(metadata)

    selected = [group["representative"] for group in groups.values()]

    if not selected and reused_count == 0:
        print("沒有待分類資料。")
        return 0

    if args.dry_run:
        preview = build_dry_run_preview(selected[: args.batch_size], taxonomy, args.model, args.max_prompt_chars)
        args.dry_run_output.parent.mkdir(parents=True, exist_ok=True)
        args.dry_run_output.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已建立請求預覽：{args.dry_run_output}")
        return 0

    if not selected and reused_count > 0:
        if reused_rows:
            write_jsonl(args.output, reused_rows)
        print(f"已從相同 Prompt 的既有結果回填 {reused_count} 筆，不需要呼叫 API。")
        return 0

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("找不到 OPENAI_API_KEY。請先設定 API 金鑰，或先使用 --dry-run 測試。")

    if args.restart:
        for path in [args.output, args.usage_output]:
            if path.exists():
                path.unlink()

    if reused_rows:
        write_jsonl(args.output, reused_rows)

    total_unique = len(selected)
    total_records = sum(len(group["records"]) for group in groups.values())
    processed_unique = 0
    processed_records = reused_count
    for index, batch in enumerate(batches(selected, args.batch_size), 1):
        batch_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{index:06d}"
        items, response = classify_batch(
            batch,
            taxonomy,
            model=args.model,
            api_key=api_key,
            max_prompt_chars=args.max_prompt_chars,
            timeout=args.timeout,
            retries=args.retries,
        )
        output_rows: list[dict[str, Any]] = []
        expanded_record_count = 0
        for record, classification in zip(batch, items):
            group = groups[str(record["prompt_hash"])]
            for metadata in group["records"]:
                output_rows.append(output_row(
                    metadata,
                    classification,
                    taxonomy_version=taxonomy["version"],
                    classifier_model=str(response.get("model") or args.model),
                    batch_id=batch_id,
                    include_prompt=args.include_prompt,
                    prompt=str(record["prompt"]),
                ))
                expanded_record_count += 1
        write_jsonl(args.output, output_rows)
        write_jsonl(args.usage_output, [
            response_usage_row(response, batch_id, args.model, len(batch), expanded_record_count)
        ])
        processed_unique += len(batch)
        processed_records += expanded_record_count
        print(
            f"已完成 {processed_unique}/{total_unique} 種唯一 Prompt，"
            f"回填 {processed_records}/{total_records + reused_count} 筆紀錄；最新批次：{batch_id}"
        )
        if args.sleep > 0 and processed_unique < total_unique:
            time.sleep(args.sleep)

    print(f"分類完成：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

