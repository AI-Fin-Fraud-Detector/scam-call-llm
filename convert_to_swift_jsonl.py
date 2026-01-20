import json
import random
from typing import List, Dict, Any, Tuple

MAX_CHARS = 4500
SEED = 42

SYSTEM_PROMPT = """You are a strict binary classification system specialized in fraud detection. Your task is to analyze a conversation log between two parties and determine if it exhibits characteristics of a scam or fraudulent intent.

**Input Format:**
The user will provide a conversation text enclosed within <conversation> tags.

**Classification Criteria:**
- Output 'True': If the conversation contains evidence of scamming, phishing, social engineering, financial fraud, or malicious intent by either party.
- Output 'False': If the conversation appears to be a normal, benign interaction without fraudulent intent.

**Output Constraints (CRITICAL):**
1. You must output EXACTLY one word: "True" or "False".
2. Do NOT output any explanation, reasoning, preamble, or punctuation.
3. Do NOT output markdown formatting (e.g., no bold, no code blocks).
4. Do NOT apologize or converse.
5. If the input is empty or unintelligible, output "False" (as the safe default) or handle strictly as per specific edge-case logic.
"""

def build_record(dialogue: str, label: int, cut_pct: int) -> Dict[str, Any]:
    user_prompt = (
        "Here is the conversation log to classify: "
        f"<conversation>\n{dialogue}\n</conversation>\n"
        'Reminder: Based on the system instructions, output ONLY "True" or "False".'
    )
    assistant = "True" if int(label) == 1 else "False"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant},
        ],
        "label": int(label),       # 保留，方便離線算 accuracy
        "cut_pct": int(cut_pct),   # 0
    }

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def write_jsonl(lines: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in lines:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote: {path} ({len(lines)} lines)")

def main():
    rng = random.Random(SEED)

    raw = read_jsonl("./syn_data/all.jsonl")

    # 1) 長度過濾 & 轉換
    ms_records = []
    for ex in raw:
        d = ex.get("dialogue", "")
        # 過濾長度 MAX_CHARS (4500)
        if isinstance(d, str) and len(d) <= MAX_CHARS:
             # 直接轉換，cut_pct=0
             record = build_record(d, ex["labels"], 0)
             ms_records.append(record)

    # 2) shuffle
    rng.shuffle(ms_records)
    test = ms_records
    
    print(f"Total processed records: {len(test)}")

    write_jsonl(test, "./syn_data/syn_test.jsonl")

if __name__ == "__main__":
    main()
