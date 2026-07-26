"""Threshold calibration and evaluation for SSCI.

Pipeline position: Stage C/D. Consumes the per-conversation SSCI traces from
``run_ssci.py``.

Alert rule (matches the production frontend)
---------------------------------------------
The frontend displays ``scam_probability`` (= confidence if the trigger decision
is scam, else 1 - confidence) and ``SSCI_SCAM_THRESHOLD`` is the threshold on it.
As a real-time monitor fires the first time the score crosses theta, the
conversation-level alert score is the running maximum over actionable triggers:

    score = max( scam_probability over triggers k >= alert_start_k )

The first trigger (6 sentences, k=1) is excluded from alerting (--alert-start-k 2)
but still feeds later triggers' agreement/stability inside run_ssci.

What it does
------------
1. Aggregate each conversation to the scam_probability alert score above.
2. Calibrate the decision threshold theta on the VALIDATION split by minimizing
   the asymmetric expected cost (default FN:FP = 2:1 -- missing a scam hurts
   twice as much as a false alarm). The structural SSCI params stay at the
   paper defaults; only theta is fit (per the agreed plan).
3. Report on the TEST split at theta*: confusion / precision / recall / F1 /
   FPR, threshold-free PR-AUC & ROC-AUC, the theta operating curve, and
   time-to-detection on caught scams.
4. Contrast against raw-model baselines (no SSCI) to show what the index buys.

Run inside the infer container:
    python -m ssci_eval.calibrate --val val --test test --fn-cost 2 --fp-cost 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .paths import DEFAULT_OUTPUT_DIR


# ----------------------------------------------------------------------------- loading
def load_convs(
    split: str, alert_start_k: int = 2, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> list[dict]:
    """Load SSCI traces.

    ``alert_start_k`` is the first trigger index that may raise an alert. The
    earlier triggers (e.g. the 6-sentence trigger at k=1) are still computed and
    still feed later triggers' agreement/stability inside run_ssci -- they are
    only excluded here as *alert candidates*, because their short-context
    confidence is too noisy to act on or to calibrate against.
    """
    path = output_dir / "ssci" / f"{split}.jsonl"
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    convs = []
    for r in rows:
        actionable = r["trace"][alert_start_k - 1:]   # drop the first (k<alert_start_k) triggers
        # Production scam score = scam_probability (conf if decision==scam else 1-conf).
        # SSCI_SCAM_THRESHOLD is the threshold on it. Sticky alert = max over the
        # call (a real-time monitor fires the first time the score crosses theta).
        probs = [t["scam_probability"] for t in actionable]
        convs.append(
            {
                "idx": r["idx"],
                "label": int(r["label"]),
                "n_triggers": len(r["trace"]),
                "n_actionable": len(actionable),
                "ns": [t["n"] for t in actionable],        # sentences seen at each trigger
                "probs": probs,                            # per-trigger scam_probability
                "decisions": [bool(t["decision"]) for t in actionable],  # raw model decisions
                "score": float(max(probs)) if probs else 0.0,
            }
        )
    return convs


# ----------------------------------------------------------------------------- metrics
def confusion(labels: np.ndarray, preds: np.ndarray) -> dict:
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "fpr": round(fpr, 4), "specificity": round(spec, 4),
    }


def cost_of(cm: dict, fn_cost: float, fp_cost: float) -> float:
    return fn_cost * cm["FN"] + fp_cost * cm["FP"]


def calibrate_threshold(convs: list[dict], fn_cost: float, fp_cost: float, cost_tol: float = 1.0) -> dict:
    """Pick theta minimizing expected cost on the validation split.

    With few positives the cost surface is jagged and its strict argmin tends to
    sit at a degenerate extreme (e.g. "alert on any scam flag"). To stay robust
    we take all thetas within ``cost_tol`` of the minimum cost and choose the
    *highest* such theta (fewest false alarms), breaking ties by F1. Set
    ``cost_tol=0`` for the strict cost-optimal threshold.
    """
    labels = np.array([c["label"] for c in convs])
    scores = np.array([c["score"] for c in convs])
    # Fine grid (not just observed scores) so a flat min-cost plateau between
    # score clusters is found, not collapsed to a single observed point.
    candidates = sorted({round(x * 0.005, 3) for x in range(201)} | set(scores.tolist()))
    evald = []
    for theta in candidates:
        cm = confusion(labels, (scores >= theta).astype(int))
        evald.append((theta, cm, cost_of(cm, fn_cost, fp_cost)))
    min_cost = min(c for _, _, c in evald)
    near = [(th, cm, c) for th, cm, c in evald if c <= min_cost + cost_tol]
    # The min-cost band can be wide with few positives. With an FN-heavy cost we
    # favor recall: take the LOWEST theta in the band (just above the normal-score
    # floor). The band [low, high] is reported so its width (uncertainty) is visible.
    theta, cm, cost = min(near, key=lambda x: (x[0], -x[1]["f1"]))
    band = (min(th for th, _, _ in near), max(th for th, _, _ in near))
    return {
        "theta": float(theta), "cm": cm, "cost": float(cost),
        "min_cost": float(min_cost), "band": [round(band[0], 4), round(band[1], 4)],
    }


def time_to_detection(convs: list[dict], theta: float) -> dict:
    """Among true scams that are caught, how many sentences until the first
    actionable trigger whose scam_probability clears theta."""
    ns = []
    for c in convs:
        if c["label"] != 1:
            continue
        for prob, n in zip(c["probs"], c["ns"]):
            if prob >= theta:
                ns.append(n)
                break
    if not ns:
        return {"caught": 0, "median_sentences": None, "mean_sentences": None}
    return {
        "caught": len(ns),
        "median_sentences": float(np.median(ns)),
        "mean_sentences": round(float(np.mean(ns)), 2),
        "p90_sentences": float(np.percentile(ns, 90)),
    }


def operating_curve(convs: list[dict], thetas: list[float]) -> list[dict]:
    """Precision/recall/FPR across a sweep of theta on the given split."""
    labels = np.array([c["label"] for c in convs])
    scores = np.array([c["score"] for c in convs])
    rows = []
    for th in thetas:
        cm = confusion(labels, (scores >= th).astype(int))
        rows.append({"theta": round(th, 3), **cm})
    return rows


def baseline_preds(convs: list[dict], mode: str) -> np.ndarray:
    """Raw-model conversation predictions, no SSCI."""
    out = []
    for c in convs:
        d = c["decisions"]
        if not d:                       # no actionable trigger -> predict normal
            out.append(0)
        elif mode == "any":
            out.append(int(any(d)))
        elif mode == "last":
            out.append(int(d[-1]))
        elif mode == "majority":
            out.append(int(sum(d) > len(d) / 2))
        else:
            raise ValueError(mode)
    return np.array(out)


def auc_scores(convs: list[dict]) -> dict:
    labels = np.array([c["label"] for c in convs])
    scores = np.array([c["score"] for c in convs])
    return {
        "pr_auc": round(float(average_precision_score(labels, scores)), 4),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 4),
    }


# ----------------------------------------------------------------------------- driver
def fmt_cm(tag: str, cm: dict, extra: str = "") -> str:
    return (
        f"  {tag:<18} P={cm['precision']:.3f} R={cm['recall']:.3f} F1={cm['f1']:.3f} "
        f"FPR={cm['fpr']:.3f}  [TP={cm['TP']} FP={cm['FP']} FN={cm['FN']} TN={cm['TN']}]{extra}"
    )


def run(args: argparse.Namespace) -> None:
    val = load_convs(args.val, args.alert_start_k, args.output_dir)
    test = load_convs(args.test, args.alert_start_k, args.output_dir)

    def dist(convs):
        pos = sum(c["label"] for c in convs)
        return f"{len(convs)} convs ({pos} scam / {len(convs) - pos} normal)"

    print("=" * 78)
    print(f"SSCI threshold calibration  (Phase 1-3, FN:FP = {args.fn_cost}:{args.fp_cost})")
    print(f"  alert from trigger k>={args.alert_start_k} (6-sentence trigger kept in history, excluded from alerting)")
    print(f"  val  = {dist(val)}")
    print(f"  test = {dist(test)}")
    print("=" * 78)

    # --- calibrate on val ---
    cal = calibrate_threshold(val, args.fn_cost, args.fp_cost)
    theta = cal["theta"]
    val_auc = auc_scores(val)
    print("\n[VAL] threshold-free:   PR-AUC={pr_auc}  ROC-AUC={roc_auc}".format(**val_auc))
    print(
        f"[VAL] chosen theta* = {theta:.4f}  (min-cost band {cal['band']}, "
        f"min expected cost = {cal['cost']:.1f})"
    )
    print(fmt_cm("val @theta*", cal["cm"]))

    # --- evaluate on test ---
    labels = np.array([c["label"] for c in test])
    scores = np.array([c["score"] for c in test])
    preds = (scores >= theta).astype(int)
    test_cm = confusion(labels, preds)
    test_auc = auc_scores(test)
    ttd = time_to_detection(test, theta)
    fa_rate = test_cm["FP"] / (test_cm["FP"] + test_cm["TN"]) if (test_cm["FP"] + test_cm["TN"]) else 0.0

    print("\n" + "-" * 78)
    print("[TEST] SSCI results")
    print("-" * 78)
    print("  threshold-free:   PR-AUC={pr_auc}  ROC-AUC={roc_auc}".format(**test_auc))
    print(fmt_cm(f"SSCI @theta={theta:.3f}", test_cm, f"  cost={cost_of(test_cm, args.fn_cost, args.fp_cost):.0f}"))
    print(f"  time-to-detection (caught scams): {ttd}")
    print(f"  false-alarm rate on normal calls: {fa_rate:.3f}")

    # --- theta operating curve on test (precision/recall/FPR tradeoff) ---
    curve = operating_curve(test, [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9])
    print("\n  theta operating curve (test):")
    for r in curve:
        print(
            f"    theta={r['theta']:<5} P={r['precision']:.3f} R={r['recall']:.3f} "
            f"F1={r['f1']:.3f} FPR={r['fpr']:.3f}  [TP={r['TP']} FP={r['FP']} FN={r['FN']}]"
        )

    # --- baselines (no SSCI) on test ---
    print("\n" + "-" * 78)
    print("[TEST] raw-model baselines (no SSCI, conversation-level)")
    print("-" * 78)
    baselines = {}
    for mode in ("any", "last", "majority"):
        bp = baseline_preds(test, mode)
        bcm = confusion(labels, bp)
        baselines[mode] = bcm
        print(fmt_cm(f"raw-{mode}", bcm, f"  cost={cost_of(bcm, args.fn_cost, args.fp_cost):.0f}"))

    # --- save ---
    report = {
        "config": {"fn_cost": args.fn_cost, "fp_cost": args.fp_cost, "val": args.val, "test": args.test},
        "theta": theta,
        "val": {"auc": val_auc, "cm": cal["cm"], "cost": cal["cost"]},
        "test": {
            "auc": test_auc, "cm": test_cm,
            "cost": cost_of(test_cm, args.fn_cost, args.fp_cost),
            "time_to_detection": ttd, "false_alarm_rate": round(fa_rate, 4),
            "operating_curve": curve, "baselines": baselines,
        },
    }
    out = args.output_dir / "calibration_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[calibrate] report saved -> {out}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSCI threshold calibration & evaluation")
    p.add_argument("--val", default="val")
    p.add_argument("--test", default="test")
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"SSCI artifact directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument("--fn-cost", dest="fn_cost", type=float, default=2.0)
    p.add_argument("--fp-cost", dest="fp_cost", type=float, default=1.0)
    p.add_argument(
        "--alert-start-k", dest="alert_start_k", type=int, default=2,
        help="first trigger index allowed to raise an alert (default 2 = exclude the "
        "6-sentence trigger's confidence from alerting/calibration, but keep it in history)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
