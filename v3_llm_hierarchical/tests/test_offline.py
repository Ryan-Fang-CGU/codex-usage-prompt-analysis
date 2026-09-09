from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

import llm_hierarchical_classifier as classifier
import summarize_classifications as summarizer


class ClassifierOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.taxonomy = classifier.read_json(BASE_DIR / "config" / "taxonomy.json")

    def test_schema_and_taxonomy_version(self) -> None:
        schema = classifier.output_schema(self.taxonomy)
        self.assertEqual(self.taxonomy["version"], classifier.VERSION)
        self.assertEqual(schema["properties"]["items"]["type"], "array")

    def test_validate_normal_multilabel_result(self) -> None:
        parsed = {
            "items": [{
                "id": "a",
                "is_internal_prompt": False,
                "domains": [
                    {"major": "生物與醫學", "minor": "肌肉", "detail": "肌肉實驗"},
                    {"major": "工程與資訊", "minor": "資料科學", "detail": "資料分析"},
                ],
                "tasks": [{"major": "資料處理與分析", "minor": "統計分析", "detail": "分析"}],
                "artifacts": [{"major": "試算表", "minor": "Excel", "detail": "表格"}],
                "technologies": ["Python"],
                "risk_flags": ["無"],
                "confidence": 0.9,
                "needs_review": False,
                "reason": "跨生物與資料分析。",
            }]
        }
        items = classifier.validate_batch_result(parsed, ["a"], self.taxonomy)
        self.assertEqual(len(items[0]["domains"]), 2)

    def test_internal_prompt_cannot_have_other_labels(self) -> None:
        parsed = {
            "items": [{
                "id": "a",
                "is_internal_prompt": True,
                "domains": [{"major": "無法判定", "minor": "Codex內部提示", "detail": ""}],
                "tasks": [], "artifacts": [], "technologies": [], "risk_flags": ["無"],
                "confidence": 1.0, "needs_review": False, "reason": "內部提示。",
            }]
        }
        with self.assertRaises(ValueError):
            classifier.validate_batch_result(parsed, ["a"], self.taxonomy)

    def test_non_internal_requires_domain_and_task(self) -> None:
        parsed = {
            "items": [{
                "id": "a", "is_internal_prompt": False,
                "domains": [], "tasks": [], "artifacts": [], "technologies": [],
                "risk_flags": ["無"], "confidence": 0.2, "needs_review": True,
                "reason": "資訊不足。",
            }]
        }
        with self.assertRaises(ValueError):
            classifier.validate_batch_result(parsed, ["a"], self.taxonomy)

    def test_summary_allocates_multilabel_tokens(self) -> None:
        records = summarizer.read_jsonl(BASE_DIR / "examples" / "demo_classified.jsonl")
        rows = summarizer.aggregate_axis(records, "domains")
        muscle = next(row for row in rows if row["minor"] == "肌肉")
        self.assertEqual(muscle["duplicated_total_tokens"], 1800)
        self.assertEqual(muscle["allocated_total_tokens"], 900.0)

    def test_dry_run_preview_is_valid_json(self) -> None:
        records = list(classifier.iter_input_jsonl(BASE_DIR / "examples" / "demo_input.jsonl"))
        preview = classifier.build_dry_run_preview(records, self.taxonomy, "test-model", 12000)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "preview.json"
            path.write_text(json.dumps(preview, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["model"], "test-model")

    def test_existing_results_build_prompt_hash_cache(self) -> None:
        completed, cache = classifier.read_existing_results(BASE_DIR / "examples" / "demo_classified.jsonl")
        self.assertEqual(completed, {"demo-001", "demo-002"})
        self.assertEqual(set(cache), {"demo-001-hash", "demo-002-hash"})


if __name__ == "__main__":
    unittest.main()

