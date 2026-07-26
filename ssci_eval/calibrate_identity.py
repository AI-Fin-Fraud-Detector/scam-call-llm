"""Per-identity threshold calibration (sensitivity analysis).

The caller identity (contact / non_contact / private) only affects SSCI's Phase-4
blend, not the model's decisions, so we reuse the cached trigger sequences and
recompute scam_probability under each identity, then calibrate a separate
threshold per identity.

IMPORTANT: this dataset has no real caller-identity labels. Each calibration
assumes the *entire* corpus is of one identity type, so these are sensitivity
estimates, not deployment-ready values.

Run inside the infer container:
    python -m ssci_eval.calibrate_identity --fn-cost 2 --fp-cost 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .const import CALLER_TYPE_CONTACT, CALLER_TYPE_NON_CONTACT, CALLER_TYPE_PRIVATE
from .ssci import compute_ssci
from .paths import DEFAULT_OUTPUT_DIR

ALERT_START_K = 2
IDENTITIES = [CALLER_TYPE_CONTACT, CALLER_TYPE_NON_CONTACT, CALLER_TYPE_PRIVATE]


def conv_scores(
    split: str, caller_type: str, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> list[dict]:
    """Conversation-level scores under a given caller identity."""
    rows = [
        json.loads(ln)
        for ln in (output_dir / "stream_infer" / f"{split}.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    out = []
    for r in rows:
        trig = [bool(x) for x in r["trigger_results"]]
        if not trig:
            continue
        probs = [compute_ssci(trig[: k + 1], caller_type=caller_type)["scam_probability"] for k in range(len(trig))]
        actionable = probs[ALERT_START_K - 1:]
        out.append({"label": int(r["label"]), "score": float(max(actionable)) if actionable else 0.0})
    return out


def confusion(labels, preds):
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=prec, recall=rec, f1=f1, fpr=fpr)


def calibrate(convs, fn_cost, fp_cost, cost_tol=1.0):
    labels = np.array([c["label"] for c in convs])
    scores = np.array([c["score"] for c in convs])
    grid = sorted({round(x * 0.005, 3) for x in range(201)} | set(scores.tolist()))
    evald = [(t, confusion(labels, (scores >= t).astype(int))) for t in grid]
    costed = [(t, cm, fn_cost * cm["FN"] + fp_cost * cm["FP"]) for t, cm in evald]
    min_cost = min(c for _, _, c in costed)
    near = [(t, cm, c) for t, cm, c in costed if c <= min_cost + cost_tol]
    theta, cm, cost = min(near, key=lambda x: (x[0], -x[1]["f1"]))
    band = (min(t for t, _, _ in near), max(t for t, _, _ in near))
    return {"theta": theta, "band": band, "min_cost": min_cost}


def run(args):
    print("=" * 90)
    print(f"Per-identity threshold calibration (FN:FP = {args.fn_cost}:{args.fp_cost}, alert from k>=2)")
    print("SENSITIVITY ANALYSIS — no real identity labels; assumes whole corpus is one identity.")
    print("=" * 90)
    print(f"{'identity':<14}{'normal-floor':<14}{'val band':<20}{'theta*':<9}"
          f"{'test P':<9}{'test R':<9}{'test F1':<9}{'FPR':<8}{'PR-AUC':<8}{'cost':<6}")
    print("-" * 90)
    report = {}
    for ct in IDENTITIES:
        val = conv_scores(args.val, ct, args.output_dir)
        test = conv_scores(args.test, ct, args.output_dir)
        cal = calibrate(val, args.fn_cost, args.fp_cost)
        theta = cal["theta"]
        labels = np.array([c["label"] for c in test])
        scores = np.array([c["score"] for c in test])
        cm = confusion(labels, (scores >= theta).astype(int))
        pr = average_precision_score(labels, scores)
        # normal floor = median score of normal conversations (test)
        floor = float(np.median(scores[labels == 0]))
        cost = args.fn_cost * cm["FN"] + args.fp_cost * cm["FP"]
        print(f"{ct:<14}{floor:<14.3f}[{cal['band'][0]:.3f},{cal['band'][1]:.3f}]{'':<6}"
              f"{theta:<9.3f}{cm['precision']:<9.3f}{cm['recall']:<9.3f}{cm['f1']:<9.3f}"
              f"{cm['fpr']:<8.3f}{pr:<8.3f}{cost:<6.0f}")
        report[ct] = {"theta": theta, "band": list(cal["band"]), "normal_floor": floor,
                      "test": {k: cm[k] for k in ("precision", "recall", "f1", "fpr", "TP", "FP", "FN", "TN")},
                      "pr_auc": round(float(pr), 4)}
    out = args.output_dir / "identity_thresholds.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-" * 90)
    print(f"saved -> {out}")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--val", default="val")
    p.add_argument("--test", default="test")
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"SSCI artifact directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument("--fn-cost", dest="fn_cost", type=float, default=2.0)
    p.add_argument("--fp-cost", dest="fp_cost", type=float, default=1.0)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
