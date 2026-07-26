"""Apply SSCI to cached streaming-inference results.

Reads the per-conversation trigger decisions produced by ``stream_infer.py`` and
replays ``compute_ssci`` trigger by trigger, recording the full sub-score trace.

The production frontend's displayed "scam score" is exactly ``compute_ssci``'s
``scam_probability`` field:

    scam_probability = confidence       if the trigger decision is "scam"  (T)
                     = 1 - confidence    if the trigger decision is "normal" (F)

and ``SSCI_SCAM_THRESHOLD`` is the threshold on that score. So we record
``scam_probability`` straight from ``compute_ssci`` (caller_type=None -> the
neutral non_contact prior, since this dataset has no caller-identity labels) and
the calibration keys off it. Triggers fire at 6, 12, 18, ... sentences; the
6-sentence trigger is kept here so it feeds later triggers' agreement/stability,
and is excluded from alerting downstream (calibrate.py --alert-start-k).

Run inside the infer container:
    python -m ssci_eval.run_ssci --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .ssci import compute_ssci
from .paths import DEFAULT_OUTPUT_DIR


def build_trace(trigger_results: list[bool], trigger_lengths: list[int]) -> list[dict]:
    """Replay SSCI trigger by trigger, recording compute_ssci's fields verbatim
    (this mirrors what production computes). ``n`` is the true cumulative
    sentence count, recorded for reference / time-to-detection."""
    trace: list[dict] = []
    for k in range(1, len(trigger_results) + 1):
        d = compute_ssci(trigger_results[:k], caller_type=None)
        n_k = trigger_lengths[k - 1] if k - 1 < len(trigger_lengths) else d["n_k"]
        trace.append(
            {
                "k": k,
                "n": n_k,
                "decision": bool(trigger_results[k - 1]),
                "evidence": d["evidence"],
                "agreement": d["agreement"],
                "stability": d["stability"],
                "raw_confidence": d["raw_confidence"],
                "confidence": d["confidence"],
                "scam_probability": d["scam_probability"],  # production scam score
            }
        )
    return trace


def summarize(label, trace: list[dict]) -> dict:
    probs = [t["scam_probability"] for t in trace]
    argmax = max(range(len(trace)), key=lambda i: probs[i])
    return {
        "label": label,
        "n_triggers": len(trace),
        "max_scam_probability": max(probs),
        "argmax_k": trace[argmax]["k"],
        "argmax_n": trace[argmax]["n"],
        "final_scam_probability": probs[-1],
        "trace": trace,
    }


def run(args: argparse.Namespace) -> None:
    in_path = args.output_dir / "stream_infer" / f"{args.split}.jsonl"
    out_dir = args.output_dir / "ssci"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}.jsonl"

    rows = [json.loads(ln) for ln in in_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = []
    for r in rows:
        trig = [bool(x) for x in r["trigger_results"]]
        if not trig:
            continue
        summary = summarize(r.get("label"), build_trace(trig, r.get("trigger_lengths", [])))
        summary["idx"] = r["idx"]
        out.append(summary)

    with out_path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[run_ssci] {args.split}: {len(out)} conversations -> {out_path}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply SSCI to cached streaming inference")
    p.add_argument("--split", default="val")
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"SSCI artifact directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
