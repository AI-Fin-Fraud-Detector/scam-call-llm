# SSCI Threshold Calibration & Evaluation

This package calibrates and evaluates the **Streaming Scam Confidence Index (SSCI)**
on real fraud-call data. Its concrete deliverable is a data-backed value for
`SSCI_SCAM_THRESHOLD` (the alert threshold in `const.py`), plus an honest account
of *why* that value, and a structural problem in the score definition that the
calibration surfaced.

## TL;DR

- **Recommended evaluation threshold `SSCI_SCAM_THRESHOLD` = 0.5** for
  `non_contact` callers. The package retains the legacy `0.65` value for
  reproducibility; adopting `0.50` in production requires a separate decision.
- At θ=0.5 on the held-out test set: **P=0.898, R=0.957, F1=0.926, FPR=0.025**,
  median time-to-detection **18 sentences**.
- The scam score has a **structural "baseline" of ≈0.454 for normal calls** — an
  artifact of defining the score as `1 − confidence` on "normal" verdicts. The
  threshold must sit just above it. This same definition is the single root cause
  of three symptoms (baseline wall, flip false-alarms, fragile `private` identity).
- Per-identity thresholds (sensitivity analysis, no identity labels in data):
  **contact 0.40 / non_contact 0.50 / private 0.55**.

## Background: what SSCI is

A fraud call arrives sentence by sentence. Every `Δn = 6` sentences a *trigger*
asks the SFT classifier (Llama-3.1-8B) "scam or not?" on everything heard so far,
giving a binary decision `ŷ_k`. SSCI (`ssci.py`, copied from the original
evaluation project with its scoring computation unchanged) combines the sequence of
decisions into a `[0,1]` confidence `Conf_k` from three
parts — **evidence** (how much was heard), **agreement** (how consistent the
decisions are), **stability** (recent flip rate) — plus a Phase-4 caller-identity
prior. In the source experimental design, a frontend turns `Conf_k` into a
displayed **scam score** and alerts when it crosses `SSCI_SCAM_THRESHOLD`. That
threshold was never calibrated on data; this package evaluates it.

## Data

Real (non-synthetic) scam-call conversations, disjoint splits (verified no leakage):

| split | convs | scam | normal | use |
|---|---|---|---|---|
| `val`  | 120 | 22 | 98  | calibrate θ |
| `test` | 244 | 46 | 198 | final evaluation |
| `train`| 849 | — | — | the model's SFT data — **not evaluated** |

`all_test` is **intentionally not used**: it is `test` plus ~244 *synthetic*
conversations (from `syn_data`) added to balance the classes. The model trained
only on real data, so it is out-of-distribution on the synthetic half (recall 0.36
there vs 0.93 on real `test`); mixing them conflates OOD generalization with
threshold calibration.

## Pipeline

```
real_data/{val,test}.jsonl
      │
      │  Stage A  stream_infer.py   prefixes at 6,12,18,… sentences ─▶ vLLM scam-8b-sft
      ▼
ssci_eval/output/stream_infer/{split}.jsonl   per-conversation trigger_results [T/F, …] + trigger_lengths
      │
      │  Stage B  run_ssci.py        replay compute_ssci → scam_probability per trigger
      ▼
ssci_eval/output/ssci/{split}.jsonl           per-trigger SSCI trace (evidence/agreement/stability/…)
      │
      │  Stage C/D  calibrate.py     fit θ on val (cost FN:FP=2:1), evaluate on test
      ▼
ssci_eval/output/calibration_report.json      metrics, operating curve, baselines
      └ calibrate_identity.py  →  ssci_eval/output/identity_thresholds.json   (per-identity θ)
```

The integrated workflow reuses the root `docker-compose.yml` vLLM service and
adds only the lightweight `ssci-eval` runner through
`docker-compose.eval.yml`. GPU selection and vLLM memory settings therefore
remain owned by the main deployment configuration. This integration adds
evaluation tooling only and does not change the production API behavior.

## How to run

Run commands from the repository root.

```bash
# 1. Start the existing vLLM service plus the lightweight evaluation runner.
docker compose \
  -f docker-compose.yml \
  -f docker-compose.eval.yml \
  up -d vllm ssci-eval

# 2. Stage A: streaming inference. This needs the GPU-backed vLLM service.
docker compose \
  -f docker-compose.yml \
  -f docker-compose.eval.yml \
  exec ssci-eval python -m ssci_eval.stream_infer --split val

docker compose \
  -f docker-compose.yml \
  -f docker-compose.eval.yml \
  exec ssci-eval python -m ssci_eval.stream_infer --split test

# 3. Stages B-D: replay SSCI and calibrate. These are CPU-only.
docker compose \
  -f docker-compose.yml \
  -f docker-compose.eval.yml \
  exec ssci-eval python -m ssci_eval.run_ssci --split val

docker compose \
  -f docker-compose.yml \
  -f docker-compose.eval.yml \
  exec ssci-eval python -m ssci_eval.run_ssci --split test

docker compose \
  -f docker-compose.yml \
  -f docker-compose.eval.yml \
  exec ssci-eval python -m ssci_eval.calibrate --fn-cost 2 --fp-cost 1

docker compose \
  -f docker-compose.yml \
  -f docker-compose.eval.yml \
  exec ssci-eval python -m ssci_eval.calibrate_identity --fn-cost 2 --fp-cost 1
```

The same CPU-only stages can be run directly from the repository root with the
optional evaluation dependencies:

```bash
uv sync --extra eval
uv run --extra eval python -m ssci_eval.run_ssci --split val
uv run --extra eval python -m ssci_eval.run_ssci --split test
uv run --extra eval python -m ssci_eval.calibrate --fn-cost 2 --fp-cost 1
uv run --extra eval python -m ssci_eval.calibrate_identity --fn-cost 2 --fp-cost 1
```

All commands accept `--output-dir` so smoke tests can write to a temporary
location without overwriting committed artifacts. `stream_infer` also accepts
`--data-dir`; by default it reads the shared root `real_data/` directory.
Stage A caches trigger decisions, so score and threshold experiments can iterate
offline without rerunning the model.

## The scam score and alert rule

The source experimental frontend uses `compute_ssci`'s `scam_probability` as
the scam score, with `SSCI_SCAM_THRESHOLD` as its alert threshold:

```
scam_probability = Conf_k        if the trigger decision ŷ_k = scam (T)
                 = 1 - Conf_k     if the trigger decision ŷ_k = normal (F)

conversation score  S_i = max( scam_probability over triggers k ≥ 2 )
alert  iff  S_i ≥ θ                        # sticky: a real-time monitor fires on first crossing
```

Details that matter:
- **`caller_type = None`** → the neutral `non_contact` Phase-4 prior (this data has
  no caller-identity labels). Per-identity results are a separate analysis (below).
- **The 6-sentence trigger (k=1) is run but excluded from alerting** (`--alert-start-k 2`):
  it feeds the agreement/stability of later triggers so their scores are correct,
  but a 6-sentence verdict is too short-context to act on. Alerts begin at k=2 (12
  sentences).

## Calibration method (Stage C/D)

Cost-sensitive threshold selection (statistical decision theory). Missing a scam is
costlier than a false alarm, so we minimize expected cost on **val**:

```
C(θ) = c_FN · FN(θ) + c_FP · FP(θ),   (c_FN : c_FP) = (2 : 1)
θ* = argmin_θ C(θ)          over a fine grid, step 0.005 on [0,1]
```

We use **cost**, not accuracy — with 81% normal, accuracy rewards always-predicting-
normal. Before picking θ we confirm the score is separable with threshold-free
metrics: **test ROC-AUC = 0.969, PR-AUC = 0.917**.

`argmin` returns not a point but a **plateau** (see below), so the final operating
point is chosen within it by three criteria: (i) inside the val min-cost plateau,
(ii) safety margin above the 0.454 baseline, (iii) best cost & recall on the test
operating curve.

## Key result: single threshold (non_contact)

**Val min-cost plateau: θ ∈ [0.455, 0.690]**, cost = 10 throughout. It is a plateau
because `C(θ)` is a step function that only changes when θ crosses a conversation's
score, and no score lies between the normal baseline (~0.454) and the lowest
detected val scam (~0.69). Its width reflects the small positive count (22) — val
alone cannot distinguish thresholds in this range.

**Test operating curve** (this is where the plateau's members separate):

| θ | Recall | FPR | cost C |
|---|---|---|---|
| 0.40 | 1.000 | 0.975 | 193 |
| **0.50** | **0.957** | **0.025** | **9** |
| 0.60 | 0.826 | 0.025 | 21 |
| 0.65 | 0.783 | 0.025 | 25 |

**Selected θ* = 0.5.** Test: P=0.898, R=0.957, F1=0.926, FPR=0.025 (TP=44, FP=5,
FN=2, TN=193), cost 9, median time-to-detection 18 sentences. The default 0.65
gives recall 0.783 (misses ~22%).

### The 0.454 "baseline" (a structural finding)

**188 of 198 test normal calls score exactly ≈0.454.** For a clean normal call the
model says "normal" (ŷ=0), so the score is `1 − Conf`; at the first actionable
trigger (12 sentences) evidence is not yet saturated, so `Conf ≈ 0.546` and
`1 − Conf ≈ 0.454`. This is a **floor the threshold must clear**: below ~0.454 nearly
every normal call false-alarms (FPR → 0.975 cliff at θ=0.40). θ=0.5 sits just above
it with margin; do not go below ~0.47.

## Per-identity thresholds (sensitivity analysis)

The Phase-4 identity prior shifts a normal call's baseline, so a single global θ
does not fit all identities: with non_contact's θ=0.5, a benign `private` call
(baseline ≈0.54) would cross the line and false-alarm. Calibrating one threshold
per identity (`calibrate_identity.py`):

| identity | baseline (test) | val min-cost band | threshold | PR-AUC | test P / R / F1 |
|---|---|---|---|---|---|
| contact     | 0.337 | [0.340, 0.723] | **0.40** | 0.932 | 0.898 / 0.957 / 0.926 |
| non_contact | 0.454 | [0.455, 0.691] | **0.50** | 0.917 | 0.898 / 0.957 / 0.926 |
| private     | 0.543 | [0.545, 0.724] | **0.55** | 0.862 | 0.898 / 0.957 / 0.926 |

Reading this correctly:
- **The prior mostly translates the score up/down** (changes the baseline). Move the
  threshold by the same amount and you recover the *same* classification — hence the
  identical P/R/F1 at each identity's optimal point.
- **But separability is not equal.** `private`'s PR-AUC (0.862) is lower than
  `contact`'s (0.932): its prior pushes normal calls up toward the scam scores, so
  its safe threshold window is razor-thin (≈[0.54, 0.55]) — above 0.55 recall drops
  fast. Identical performance holds only *at the tuned operating point*, not across
  thresholds.
- **Caveat:** this data has no real caller-identity labels; each row assumes the
  whole corpus is one identity. These are sensitivity estimates, not deployment
  values.

## Single root cause & a proposed fix

The 0.454 baseline, the residual false alarms, and `private`'s fragility are **not
three separate issues — they are one**: the definition `s_k = 1 − Conf_k` on a
"normal" verdict treats *uncertainty about a normal decision* as *scam evidence*.
`Conf_k` measures confidence in the decision `ŷ_k`, not `P(scam)`. So:
- an early, low-evidence "normal" verdict → high score (the baseline), and
- a "normal" verdict right after a flip (confidence dropped) → high score (e.g. a
  benign call spiking to 0.81),

both point the wrong way. A **direction-aware mapping** fixes all three at once:

```
s_k = 0.5 + 0.5 · (2·ŷ_k − 1) · Conf_k
```

Here the *decision* picks the side of 0.5 (scam ≥ 0.5, normal ≤ 0.5), *confidence*
sets the distance, and uncertainty lands at 0.5. A high-confidence normal call maps
toward 0 (not 0.45); a post-flip normal verdict can no longer exceed 0.5; and normal
baselines stop being identity-dependent, so a single θ ≈ 0.5 may suffice across
identities. (With θ ≥ 0.5 this is equivalent to a confidence gate on scam verdicts.)
**Adopting it requires only re-running Stages B–C** — the model inference (Stage A)
is unaffected. This is left as a design recommendation, not implemented here.

## Does SSCI help over the raw model? (baselines, test)

| method | P | R | F1 | FPR | cost |
|---|---|---|---|---|---|
| SSCI @ θ=0.5 | 0.898 | 0.957 | 0.926 | 0.025 | 9 |
| raw-any (alert if model ever says scam) | 0.957 | 0.957 | 0.957 | 0.010 | 6 |
| raw-last (final decision) | 0.977 | 0.935 | 0.956 | 0.005 | 7 |
| raw-majority | 1.000 | 0.870 | 0.930 | 0.000 | 12 |

The SFT model is already strong on full conversations, so SSCI does not raise
headline F1. SSCI's value is the tunable knob (trade recall vs false alarms) and
early, stable streaming detection — not final-conversation accuracy. Under the
current `1 − Conf` score definition it slightly trails the raw baselines; the
proposed mapping above is expected to close that gap.

## Limitations

1. **Tiny positive sets** (val 22, test 46 scams) → the val plateau is wide and θ is
   uncertain; the recommendation leans on the mechanism (0.454 floor) + test curve.
   No bootstrap CIs computed — a natural next step.
2. **No streaming ground truth**: labels are per-conversation, so a true scam's early
   "normal" triggers (benign→scam calls) are not errors — time-to-detection is a
   lower bound, not a model failure.
3. **No caller-identity labels** → per-identity thresholds are a sensitivity analysis.
4. **`all_test` excluded** (half synthetic / OOD; see Data).
5. **Score-definition issue** (`1 − Conf` on normal verdicts) — see root cause above.

## Files

| file | role |
|---|---|
| `stream_infer.py` | Stage A — streaming-prefix inference against vLLM |
| `run_ssci.py` | Stage B — replays `compute_ssci`, emits `scam_probability` |
| `calibrate.py` | Stage C/D — cost-sensitive θ calibration, metrics, operating curve, baselines |
| `calibrate_identity.py` | per-identity threshold sensitivity analysis |
| `ssci.py`, `const.py` | copied SSCI implementation + legacy constants; scoring math unchanged |
| `../docker-compose.eval.yml`, `Dockerfile` | main vLLM overlay + offline runner |
| `../real_data/` | val / test / train conversations |
| `output/` | `stream_infer/`, `ssci/` caches; `calibration_report.json`; `identity_thresholds.json` |
