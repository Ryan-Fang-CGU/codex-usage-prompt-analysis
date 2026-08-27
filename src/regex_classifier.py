"""Pure-regex multi-label classifier used by the public Codex usage analysis.

No model inference or fine-tuning is performed by this module.
"""

from __future__ import annotations

import argparse
import json
import re
import sys


DOMAIN_PATTERNS = {
    "醫學領域": re.compile(
        r"醫療|醫學|臨床|病人|病患|護理|疾病|診斷|治療|藥物|藥品|手術|醫院|健康|生醫|"
        r"生物醫學|解剖|生理|病理|基因|蛋白質|細胞|檢驗|影像醫學|放射|復健|長照|"
        r"流行病|公共衛生|營養|獸醫|medical|clinical|patient|nursing|diagnos|treatment|"
        r"disease|hospital|healthcare|biomed|genetic|\bgene\b|protein|pathology|pharma|\bdrug\b|"
        r"surgery|epidemiolog|radiolog|therapy|nutrition|veterinar",
        re.I,
    ),
    "工程領域": re.compile(
        r"工程|程式|軟體|硬體|演算法|資料庫|網站|系統架構|機械|電機|電子|土木|化工|材料|"
        r"資工|資訊工程|人工智慧|機器學習|深度學習|網路|資安|半導體|電路|控制系統|製造|"
        r"機器人|物聯網|android|\bios\b|\bapi\b|python|javascript|typescript|\bjava\b|c\+\+|"
        r"\bsql\b|docker|kubernetes|\bcode\b|programming|software|hardware|algorithm|database|"
        r"frontend|backend|debug|github|\blinux\b|\bwindows\b|\bserver\b|\bcloud\b|robotics|"
        r"semiconductor|circuit",
        re.I,
    ),
    "商業領域": re.compile(
        r"商業|商管|企業管理|經營管理|營運管理|行銷|財務|金融|會計|經濟|企業|人力資源|投資|"
        r"股票|銷售|顧客|客戶|供應鏈|創業|品牌|預算|採購|廣告|電商|business|marketing|finance|"
        r"financial|accounting|economics|enterprise management|business management|sales|customer|"
        r"supply chain|investment|\bstock\b|budget|\bbrand\b|advertising|e-commerce|commerce|"
        r"procurement|human resources",
        re.I,
    ),
}

TASK_PATTERNS = {
    "編製PPT": re.compile(r"\bpptx?\b|power\s*point|簡報|投影片|幻燈片|slide\s*deck|\bslides?\b|簡報稿", re.I),
    "撰寫文檔": re.compile(
        r"\bdocx?\b|\bword\b|文檔|文件|報告|計畫書|企劃書|摘要|撰寫|改寫|潤飾|整理成|"
        r"翻譯|譯成|電子郵件|郵件|公文|履歷|論文|文章|新聞稿|講稿|腳本|文案|紀錄|"
        r"manuscript|\bwrite\b|\bdraft\b|summari[sz]e|\bsummary\b|document|\breport\b|proposal|"
        r"\barticle\b|\bpaper\b|proofread|translate|rewrite|email|curriculum vitae|\bcv\b",
        re.I,
    ),
    "撰寫程式": re.compile(
        r"程式|程式碼|代碼|寫碼|除錯|偵錯|開發|實作|重構|函式|套件|資料庫|網站|前端|後端|"
        r"演算法|android|\bios\b|\bapi\b|python|javascript|typescript|\bjava\b|c\+\+|\bc#\b|"
        r"\bsql\b|html|css|docker|kubernetes|\bcode\b|coding|programming|debug|refactor|implement|"
        r"function|class|repository|github|compile|runtime error|unit test|test case|script",
        re.I,
    ),
    "問題回答": re.compile(
        r"如何|怎麼|為什麼|什麼是|是否|哪一|比較|解釋|說明|請問|回答|評估|判斷|分析|"
        r"建議|告訴我|列出|找出|查詢|驗證|檢查|評分|how\b|what\b|why\b|which\b|whether|"
        r"explain|answer|evaluate|assess|analy[sz]e|recommend|identify|compare|check|review|score",
        re.I,
    ),
}

INTERNAL_PATTERNS = [
    re.compile(r"^Reply with OK only\.?$", re.I),
    re.compile(r"You are a helpful assistant\. You will be presented with a user prompt, and your job is to provide a short title", re.I),
    re.compile(r"You are performing a CONTEXT CHECKPOINT COMPACTION", re.I),
]


def clean_classification_text(text: str) -> tuple[str, bool]:
    text = text or ""
    internal = any(pattern.search(text[:1200]) for pattern in INTERNAL_PATTERNS)
    if "Another language model started to solve this problem" in text:
        marker = "Here is the summary produced by the other language model"
        if marker in text:
            text = text.split(marker, 1)[1]
    for marker in ("# My request for Codex:", "My request for Codex:", "User prompt:"):
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    if text.lstrip().startswith("Document summary:") and "\n\nChunks:" in text:
        text = text.split("\n\nChunks:", 1)[0]
    if len(text) > 16000:
        text = text[:10000] + "\n...\n" + text[-4000:]
    return text, internal


def classify(text: str) -> dict[str, object]:
    normalized, internal = clean_classification_text(text)
    if internal:
        return {"domains": [], "tasks": [], "codex_internal_prompt": True}
    domains = [name for name, pattern in DOMAIN_PATTERNS.items() if pattern.search(normalized)]
    tasks = [name for name, pattern in TASK_PATTERNS.items() if pattern.search(normalized)]
    return {
        "domains": domains or ["其他／未明領域"],
        "tasks": tasks or ["其他任務"],
        "codex_internal_prompt": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one Prompt with the public regex rules.")
    parser.add_argument("text", nargs="?", help="Prompt text; reads stdin when omitted")
    args = parser.parse_args()
    text = args.text if args.text is not None else sys.stdin.read()
    print(json.dumps(classify(text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
