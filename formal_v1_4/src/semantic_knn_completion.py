"""Complete formal-v1.4 labels with semantic embeddings, never regex rules."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from formal_v14_batch_classifier import compact_view, iter_unique_prompts, load_env


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent


def load_labels(paths: list[Path], taxonomy: dict[str, list[str]]) -> dict[str, dict]:
    labels: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row["main_category"] in taxonomy and row["sub_category"] in taxonomy[row["main_category"]]:
                    labels.setdefault(row["prompt_hash"], row)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return labels


def load_embedding_cache(path: Path) -> dict[str, np.ndarray]:
    cached: dict[str, np.ndarray] = {}
    if not path.exists():
        return cached
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            cached[row["prompt_hash"]] = np.asarray(row["embedding"], dtype=np.float32)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return cached


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def embed_batch(texts: list[str], *, api_url: str, api_key: str, model: str, timeout: int) -> tuple[list[list[float]], dict]:
    body = json.dumps({"model": model, "input": texts}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(api_url, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json; charset=utf-8"
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    ordered = sorted(payload["data"], key=lambda item: int(item.get("index", 0)))
    return [item["embedding"] for item in ordered], payload.get("usage") or {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "drive_verified_20260917")
    parser.add_argument("--taxonomy", type=Path, default=BASE / "成果" / "v1.4_既有大項歸位" / "分類表_v1.4.json")
    parser.add_argument("--labels", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--usage-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--references-per-label", type=int, default=20)
    parser.add_argument("--neighbors", type=int, default=7)
    parser.add_argument("--prompt-max-chars", type=int, default=350)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    load_env(ROOT / "第三版本_階層式語意分類" / "第三版本_階層式語意分類.env")
    load_env(ROOT / "第三版本_階層式語意分類" / ".env")
    api_key = os.environ.get("LOCAL_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        raise SystemExit("找不到 LOCAL_LLM_API_KEY")
    chat_url = os.environ.get("LOCAL_LLM_API_URL") or "https://air.cgu.edu.tw/cgullmapi/v1/chat/completions"
    api_url = chat_url.rsplit("/chat/completions", 1)[0] + "/embeddings"
    model = "bge-m3:latest"

    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    prompts = iter_unique_prompts(args.input_dir)
    by_hash = {row["prompt_hash"]: row for row in prompts}
    labels = load_labels(args.labels, taxonomy)
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for digest, row in labels.items():
        if digest in by_hash:
            grouped[(row["main_category"], row["sub_category"])].append(digest)

    rng = random.Random(20260917)
    reference_hashes: list[str] = []
    for key in sorted(grouped):
        values = grouped[key]
        rng.shuffle(values)
        reference_hashes.extend(values[: args.references_per_label])
    pending = [row for row in prompts if row["prompt_hash"] not in labels]
    if args.limit is not None:
        pending = pending[:args.limit]
    target_hashes = [row["prompt_hash"] for row in pending]
    wanted = list(dict.fromkeys(reference_hashes + target_hashes))
    cache = load_embedding_cache(args.cache)
    usage_total = {"prompt_tokens": 0, "total_tokens": 0, "requests": 0}
    missing_embeddings = [digest for digest in wanted if digest not in cache]
    print(json.dumps({"labeled": len(labels), "pending": len(pending), "references": len(reference_hashes), "to_embed": len(missing_embeddings)}, ensure_ascii=False), flush=True)
    completed_embeddings = 0
    chunks = [missing_embeddings[i:i + args.batch_size] for i in range(0, len(missing_embeddings), args.batch_size)]
    def run_chunk(hashes: list[str]):
        texts = [compact_view(by_hash[digest]["prompt"], args.prompt_max_chars)[0] for digest in hashes]
        last_error = None
        for attempt in range(1):
            try:
                vectors, usage = embed_batch(texts, api_url=api_url, api_key=api_key, model=model, timeout=args.timeout)
                return hashes, vectors, usage
            except Exception as exc:
                last_error = exc
                if attempt < 0:
                    time.sleep(1)
        raise last_error
    # Keep the queue full instead of waiting for the slowest request in a wave.
    # Each completed chunk is checkpointed immediately.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        chunk_iter = iter(chunks)
        active = {pool.submit(run_chunk, chunk): chunk for chunk in [next(chunk_iter, None) for _ in range(args.workers)] if chunk}
        next_report = 64
        while active:
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                failed_chunk = active.pop(future)
                try:
                    hashes, vectors, usage = future.result()
                except Exception:
                    # A small set of long inputs can repeatedly trigger a 500.
                    # Split a failed batch into singletons; retry a singleton
                    # without aborting the already-checkpointed full run.
                    retry_chunks = [[digest] for digest in failed_chunk] if len(failed_chunk) > 1 else [failed_chunk]
                    time.sleep(2)
                    for retry_chunk in retry_chunks:
                        active[pool.submit(run_chunk, retry_chunk)] = retry_chunk
                    continue
                rows = []
                for digest, vector in zip(hashes, vectors):
                    cache[digest] = np.asarray(vector, dtype=np.float32)
                    rows.append({"prompt_hash": digest, "embedding": vector})
                append_jsonl(args.cache, rows)
                usage_total["requests"] += 1
                usage_total["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                usage_total["total_tokens"] += int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
                completed_embeddings += len(hashes)
                chunk = next(chunk_iter, None)
                if chunk:
                    active[pool.submit(run_chunk, chunk)] = chunk
            if completed_embeddings >= next_report or not active:
                print(json.dumps({"embedded": completed_embeddings, "total": len(missing_embeddings), "usage": usage_total}, ensure_ascii=False), flush=True)
                next_report = completed_embeddings + 64

    ref_matrix = np.stack([cache[digest] for digest in reference_hashes])
    ref_matrix /= np.maximum(np.linalg.norm(ref_matrix, axis=1, keepdims=True), 1e-12)
    completed_output = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            try:
                completed_output.add(json.loads(line)["prompt_hash"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    output_rows = []
    for index, row in enumerate(pending, 1):
        digest = row["prompt_hash"]
        if digest in completed_output:
            continue
        vector = cache[digest]
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        similarities = ref_matrix @ vector
        k = min(args.neighbors, len(reference_hashes))
        nearest = np.argpartition(similarities, -k)[-k:]
        votes: dict[tuple[str, str], float] = defaultdict(float)
        internal_votes: dict[tuple[str, str], float] = defaultdict(float)
        for ref_index in nearest:
            source = labels[reference_hashes[int(ref_index)]]
            key = (source["main_category"], source["sub_category"])
            weight = max(0.001, float(similarities[int(ref_index)])) ** 4
            votes[key] += weight
            if source.get("is_internal_prompt"):
                internal_votes[key] += weight
        winner = max(votes, key=votes.get)
        confidence = votes[winner] / max(sum(votes.values()), 1e-12)
        output_rows.append({
            "prompt_hash": digest, "occurrences": row["occurrences"], "taxonomy_version": "formal-v1.4",
            "model": "bge-m3:latest semantic-kNN over Luna/gpt-oss labels", "id": row["id"],
            "main_category": winner[0], "sub_category": winner[1],
            "is_internal_prompt": internal_votes[winner] >= votes[winner] * 0.5,
            "needs_review": confidence < 0.55, "confidence": round(confidence, 6),
        })
        if len(output_rows) >= 200:
            append_jsonl(args.output, output_rows)
            output_rows.clear()
        if index % 500 == 0:
            print(json.dumps({"classified": index, "pending": len(pending)}, ensure_ascii=False), flush=True)
    append_jsonl(args.output, output_rows)
    report = {"method": "semantic_embedding_knn_no_regex", "model": model, "labeled_reference_count": len(labels),
              "balanced_reference_count": len(reference_hashes), "classified": len(pending), "api_usage": usage_total}
    args.usage_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
