"""Streaming-prefix inference for SSCI evaluation.

Replays each conversation the way the live system sees it: a call grows turn by
turn, and every DELTA_N sentences a *trigger* fires that asks the SFT model
"scam or not?" on everything heard so far. Concretely, for a conversation with
sentences s_1..s_N we query the model on the cumulative prefixes

    s_1..s_6,  s_1..s_12,  s_1..s_18,  ...,  s_1..s_N

producing one binary decision per trigger. The ordered list of decisions
(``trigger_results``) is exactly what ``ssci.compute_ssci`` consumes downstream.

The model, prompt, and decoding settings mirror the production app
(scam-call-llm/app.py) so the binary decisions match what the deployed
classifier would emit. Results are cached to JSONL so the GPU pass runs once and
SSCI/threshold calibration can iterate offline.

Run inside the infer container (see docker-compose.yml):
    python -m ssci_eval.stream_infer --split val
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import httpx

from .const import DELTA_N
from .paths import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR

# --- Prompt / model (must stay byte-identical to the trained classifier) ------
MODEL_NAME = "scam-8b-sft"

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

USER_TEMPLATE = (
    "Here is the conversation log to classify: <conversation>\n"
    "{conversation}\n"
    "</conversation>\n"
    'Reminder: Based on the system instructions, output ONLY "True" or "False".'
)

_CONV_RE = re.compile(r"<conversation>(.*?)</conversation>", re.DOTALL)


def extract_sentences(user_content: str) -> list[str]:
    """Pull the dialogue body out of the templated user message and split it
    into sentences (one dialogue turn per line, e.g. ``caller: ...``)."""
    m = _CONV_RE.search(user_content)
    body = m.group(1) if m else user_content
    return [ln.strip() for ln in body.strip().split("\n") if ln.strip()]


def build_prefixes(
    n_sentences: int, start_n: int, delta_n: int, include_final_partial: bool
) -> list[int]:
    """Cumulative trigger lengths: start_n, start_n+delta_n, ... plus the full call.

    ``start_n`` is the warm-up: the first trigger fires only after this many
    sentences (e.g. 12 to skip the noisy 6-sentence trigger), then every delta_n
    sentences. The final partial chunk (when n_sentences is not aligned to the
    grid) is appended so the last trigger always reflects the entire conversation
    -- "run until all done". Conversations shorter than start_n yield a single
    trigger on the whole call."""
    lengths = list(range(start_n, n_sentences + 1, delta_n))
    if include_final_partial and (not lengths or lengths[-1] != n_sentences) and n_sentences > 0:
        lengths.append(n_sentences)
    return lengths


def coerce_decision(raw: str | None) -> bool:
    """True only on an exact 'True' (matches app.py's strict coercion)."""
    return (raw or "").strip() == "True"


async def classify_prefix(
    client: httpx.AsyncClient, vllm_url: str, conversation: str, retries: int = 3
) -> bool:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(conversation=conversation)},
        ],
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = await client.post(vllm_url, json=payload)
            r.raise_for_status()
            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return coerce_decision(content)
        except Exception as e:  # noqa: BLE001 - retry on any transient failure
            last_err = e
            await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"inference failed after {retries} retries: {last_err}")


async def process_conversation(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    vllm_url: str,
    idx: int,
    row: dict,
    start_n: int,
    delta_n: int,
    include_final_partial: bool,
) -> dict | None:
    user_content = next(
        (m["content"] for m in row.get("messages", []) if m.get("role") == "user"), ""
    )
    sentences = extract_sentences(user_content)
    if not sentences:
        return None

    prefix_lengths = build_prefixes(len(sentences), start_n, delta_n, include_final_partial)

    async def one(plen: int) -> bool:
        conv_text = "\n".join(sentences[:plen])
        async with sem:
            return await classify_prefix(client, vllm_url, conv_text)

    decisions = await asyncio.gather(*(one(p) for p in prefix_lengths))

    return {
        "idx": idx,
        "label": int(row.get("label")) if row.get("label") is not None else None,
        "n_sentences": len(sentences),
        "trigger_lengths": prefix_lengths,            # cumulative sentence count per trigger
        "trigger_results": [bool(d) for d in decisions],  # flat list for compute_ssci
    }


async def run(args: argparse.Namespace) -> None:
    in_path = args.data_dir / f"{args.split}.jsonl"
    out_dir = args.output_dir / "stream_infer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"{args.split}.jsonl"

    rows = [json.loads(ln) for ln in in_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[stream_infer] {in_path.name}: {len(rows)} conversations -> {out_path}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        tasks = [
            process_conversation(
                client, sem, args.vllm_url, i, row, args.start_n, args.delta_n,
                args.include_final_partial,
            )
            for i, row in enumerate(rows)
        ]
        done = 0
        for coro in asyncio.as_completed(tasks):
            res = await coro
            done += 1
            if res is not None:
                results.append(res)
            if done % 20 == 0 or done == len(tasks):
                print(f"  ...{done}/{len(tasks)} conversations done", flush=True)

    results.sort(key=lambda r: r["idx"])
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_trig = sum(len(r["trigger_results"]) for r in results)
    print(
        f"[stream_infer] wrote {len(results)} conversations, {n_trig} total triggers to {out_path}",
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Streaming-prefix inference for SSCI evaluation")
    p.add_argument("--split", default="val", help="real_data split: val / test")
    p.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"directory containing train/val/test JSONL files (default: {DEFAULT_DATA_DIR})",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"SSCI artifact directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--vllm-url",
        dest="vllm_url",
        default=os.environ.get("VLLM_URL", "http://vllm:8000/v1/chat/completions"),
    )
    p.add_argument("--delta-n", dest="delta_n", type=int, default=DELTA_N)
    p.add_argument(
        "--start-n", dest="start_n", type=int, default=DELTA_N,
        help="sentences before the first trigger (default 6). The first trigger is kept "
        "so it feeds later triggers' agreement/stability; excluding its confidence from "
        "alerting is done downstream in calibrate.py (--alert-start-k).",
    )
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--limit", type=int, default=0, help="only process first N conversations (smoke test)")
    p.add_argument(
        "--no-final-partial",
        dest="include_final_partial",
        action="store_false",
        help="do not add a final trigger for the leftover (<delta_n) tail of the call",
    )
    p.add_argument("--out", default="", help="output JSONL path (default: <output-dir>/stream_infer/<split>.jsonl)")
    return p.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(run(parse_args(sys.argv[1:])))
