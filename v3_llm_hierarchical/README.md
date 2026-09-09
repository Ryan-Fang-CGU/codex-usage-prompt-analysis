# 第三版本：LLM 階層式語意分類

這一版已完全移除 Prompt 的正則與關鍵字分類。每筆 Prompt 由語言模型依語意同時判斷四個維度：

1. 領域：例如生物與醫學、工程與資訊、自然科學、法律。
2. 任務：例如問答、文件撰寫、資料分析、Debug、前端、後端。
3. 產出物：例如 Word、Excel、PowerPoint、Canva、程式碼、網頁。
4. 技術：例如 Python、MySQL、React、GitHub。

同一 Prompt 可以跨領域、多任務、多產出物。Codex 內部提示會獨立標記，且不歸入其他類別。

## 和前兩版的差異

- 不呼叫第一版的 `classify()`，也不使用正則判斷 Prompt 類別。
- 使用大類 `major`、小類 `minor`、補充主題 `detail` 三層結果。
- 不要求逐筆人工標注；模型會提供 `confidence` 與 `needs_review`。
- 保留可中斷續跑功能，已完成的 Prompt 不會重複花費分類 Token。
- 完全相同的 Prompt 只呼叫模型一次，再把結果回填到全部使用紀錄。
- API 使用 Token 與原始 Codex 使用 Token 分開統計。
- 多標籤彙總同時提供重複計算 `duplicated_*` 與均分計算 `allocated_*`。

## 重要統計限制

不進行人工標注是可以的，但這時不能宣稱模型分類的 Accuracy、Precision、Recall 或 F1。模型自行提供的 `confidence` 也不是真實正確率。

可報告的品質指標包括：

- 高、中、低信心比例。
- `needs_review` 比例。
- 無法判定比例。
- 若日後重複分類，可計算兩次分類的一致率。
- 少量人工抽查可作為品質稽核，但不是執行本流程的必要條件。

## 檔案

```text
config/taxonomy.json              分類大類、小類與判斷政策
src/llm_hierarchical_classifier.py 讀取資料並呼叫模型分類
src/summarize_classifications.py    依領域、任務、產出物及技術彙總 Token
examples/demo_input.jsonl           可自行測試的輸入範例
examples/demo_classified.jsonl      不需 API 即可測試彙總的範例結果
成果/                               正式執行時產生，不應直接公開 Prompt 原文
```

## 先做離線測試

不呼叫 API，只建立請求預覽：

```powershell
python src/llm_hierarchical_classifier.py `
  --input-jsonl examples/demo_input.jsonl `
  --limit 2 `
  --dry-run
```

測試彙總程式：

```powershell
python src/summarize_classifications.py `
  --input examples/demo_classified.jsonl `
  --output-dir 成果/demo_彙總
```

## 正式分類

先在目前的 PowerShell 工作階段設定 API 金鑰；不要把金鑰寫進程式、Excel 或 GitHub：

```powershell
$env:OPENAI_API_KEY = "你的金鑰"
```

先用 20 筆確認分類：

```powershell
python src/llm_hierarchical_classifier.py --limit 20 --batch-size 8
```

確認後分類其餘資料：

```powershell
python src/llm_hierarchical_classifier.py --batch-size 8
```

預設會從既有 `data` 資料夾讀取 `.clean.json` 與 ZIP；結果寫入：

- `成果/prompt_classifications.jsonl`
- `成果/classification_api_usage.jsonl`

程式預設不把 Prompt 原文寫入分類結果。重新執行時會略過已完成的 id；只有明確使用 `--restart` 才會重新分類。

`--limit 20` 代表先分類 20 種不重複的 Prompt。若相同 Prompt 在紀錄中出現多次，只消耗一次分類 API，之後會自動回填到每筆紀錄，以保留原始 Token 與學系統計。

分類完成後執行：

```powershell
python src/summarize_classifications.py
```

會產生領域、任務、產出物、技術、領域×任務 CSV，以及分類總覽 JSON。

## API 實作

分類器使用 OpenAI Responses API、Structured Outputs JSON Schema 與 `store=false`。分類表會完整傳入模型，模型輸出若不符合既有 major/minor、遺漏 id，或將 Codex 內部提示混入其他類別，程式會判定該批無效並自動重試。

目前預設模型為 `gpt-5-mini`，也可以透過 `--model` 或 `OPENAI_MODEL` 指定其他支援 Structured Outputs 的模型。正式大量執行前，應確認帳號可用模型及最新費率。

官方介面說明：https://developers.openai.com/api/reference/python/resources/responses/methods/create

