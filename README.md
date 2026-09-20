# Codex Prompt 使用分類：正式版 v1.4 與歷史驗證

## 正式交接版本（2026-09-21）

目前正式分析採用 v1.4「單一大項＋單一小項」混合式語意分類，不使用正則表達式判定正式類別，也沒有進行 fine-tuning。完整資料涵蓋23個ZIP、120,543筆去重請求；264筆沒有非空白Prompt，另外120,279筆請求均已連結正式類別。相同Prompt去重後共有33,666個語意標籤。

正式交接檔案不含Prompt原文、回覆內容、帳號、Email、學號或API金鑰：

```text
handoff/
  request_id_category.csv        # 120,543筆；兩欄，直接和金額表用request_id串接
  prompt_sha256_category.csv     # 33,666筆；兩欄，無request_id時用內容SHA-256串接
  processing_exceptions.csv      # 空白Prompt、合成ID、解析／重複／未分類狀態及原因
  pipeline_audit_summary.json    # 全量處理數量與例外定義
formal_v1_4/
  taxonomy/classification_v1.4.json
  src/formal_v14_batch_classifier.py
  src/semantic_knn_completion.py
  src/export_collaborator_handoff.py
```

`category`欄格式為`大項 > 小項`。`request_id_category.csv`保留全部去重請求；空白Prompt標為`未分類 > 空白 Prompt`，沒有默默丟棄。23筆來源紀錄沒有request_id，稽核時使用`archive.zip:member.json`合成識別碼；若另一份資料沒有相同合成ID，應改用`prompt_sha256_category.csv`串接。

### 正式分類方法

- 第一階段以正式v1.4分類表進行語意分類，保留30,129個Luna標籤與13個gpt-oss標籤。
- 第二階段對其餘3,524個唯一Prompt使用BGE-M3語意向量，依平衡抽取的既有語意標籤進行k近鄰補全。
- 最終每個非空白Prompt只有一個大項及一個小項；內部提示仍保留`is_internal_prompt`品質旗標，但不建立獨立用途大項。
- 正式結果沒有剩餘的非空白未分類資料。API過程中的暫時逾時若最後取得分類，不算最終失敗。

### 全量稽核結果

| 狀態 | 筆數 |
|---|---:|
| ZIP | 23 |
| JSON紀錄 | 120,543 |
| 去重請求 | 120,543 |
| 已分類請求 | 120,279 |
| 空白Prompt | 264 |
| 非空白但未分類 | 0 |
| JSON解析失敗 | 0 |
| 損壞ZIP | 0 |
| 重複request_id | 0 |
| 缺少request_id而使用合成ID | 23 |

以下「正則分類與混合參考驗證」是早期研究版本，保留供方法比較，不是目前全量正式分類器。

本專案使用純正則表達式與關鍵字規則，將 Codex Prompt 分成多個領域與任務類別，並以「人工金標＋ChatGPT／豆包逐格共識」建立混合參考標注，評估規則分類器的表現。

## 對外主要結果

對外統一報告的指標名稱為「混合參考一致度」，不是獨立人工測試集上的真實正確率。

| 指標 | 結果 |
|---|---:|
| 混合參考一致度 | 95.1% |
| 95% 信賴區間 | 94.2%–96.0% |
| Precision | 97.2% |
| Recall | 82.3% |
| F1 | 89.1% |
| 可評估 Prompt | 435 |
| 可評估標籤格 | 2,798 |

信賴區間以 Prompt 為叢集進行 10,000 次 bootstrap，保留同一 Prompt 內多個標籤的相依性。

## 為什麼稱為「一致度」

混合參考包含人工判定及兩個 AI 完全一致時產生的共識標籤。AI 共識可增加可評估資料量，但兩個 AI 仍可能一起犯錯，而且一致案例通常較容易分類。因此 95.1% 表示規則輸出與目前混合參考標注的吻合程度，不等同於母體中的真實正確率。

目前人工金標的探索性結果為 68.2%，95% Prompt 層級 bootstrap 區間為 58.5%–76.1%；人工樣本量仍不足以作為全校 Prompt 的正式外推結果。

## 分類方法

- 不使用 fine-tuning。
- 不在分類階段呼叫大型語言模型。
- 使用 Python 正則表達式與關鍵字比對。
- 同一 Prompt 可以同時屬於多個領域與多個任務。
- Codex 內部提示獨立分類；一旦判為內部提示，不再歸入其他領域或任務。

第一版分類範圍：

- 領域：醫學、工程、商業、其他／未明。
- 任務：編製 PPT、撰寫文檔、撰寫程式、問題回答、其他任務。
- 獨立標籤：Codex 內部提示。

「數學領域」及「教材編寫」已存在於標注資料，但第一版正則分類器尚未輸出這兩個標籤，因此不納入第一版正確性計算。

## 資料夾

```text
annotations/
  final_labels_deidentified.csv
  cell_labels_deidentified.csv
  雙AI共識與人工標注_去識別.xlsx
reports/
  雙AI共識與人工標注_統計信度與正確性評估.xlsx
results/
  metrics_summary.json
src/
  regex_classifier.py
  evaluate_mixed_reference.py
```

公開標注資料不包含 Prompt 原文、對話背景、學號、帳號、Email、本機路徑、API 金鑰或原始 Token 使用紀錄。`sample_id` 使用公開版重新編排的 P0001–P0500，只能在本專案公開資料表之間對照。

## 重算結果

需要 Python 3.10 以上，無第三方套件依賴：

```bash
python src/evaluate_mixed_reference.py \
  --input annotations/cell_labels_deidentified.csv \
  --output results/metrics_summary_recomputed.json
```

## 統計限制

- 500 筆 Prompt 為目的式分層樣本，不是全校 Prompt 的簡單隨機樣本。
- 雙 AI 共識是 silver label，不是人工真值。
- 尚待人工裁決的分歧格較難；只評估共識格可能產生 easy-case bias。
- 前 16 筆若曾用於調整正則規則，就不能視為完全獨立的最終測試集。
- 正式對外使用時，建議同時呈現「混合參考一致度」名稱、95% 信賴區間與上述限制。
