# significance-gate (`siggate`)

**The vendor-neutral significance gate for AI evals. Most eval tools show a number moving; none tell you whether the move is real. siggate does — in CI.**

[![CI](https://github.com/yongzhe2160cs/significance-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/yongzhe2160cs/significance-gate/actions/workflows/ci.yml)
&nbsp;[![stats engine: deltagate](https://img.shields.io/badge/stats%20engine-deltagate-blue)](https://github.com/yongzhe2160cs/eval-reliability)

You bump a prompt, swap a checkpoint, change a decoding config. The eval moves
`+1.8%`. Ship it? Every dashboard, harness, and leaderboard will happily render
that `+1.8%` as a green number. None of them answer the only question that
matters:

> **Is `+1.8%` a real improvement, or is it noise?**

`siggate` answers it, on every eval run, as a CI check:

```
🟡 UNDERPOWERED: +1.8% is within noise (p=0.34); you'd need ~420 more samples to call it.
```

Three verdicts, one line each:

| | verdict | meaning | what to do |
|---|---|---|---|
| 🟢 | **REAL** | the change survives the statistics | ship it |
| ⚪ | **NOISE** | you had the power to find an effect; there isn't one | don't ship; it's not a change |
| 🟡 | **UNDERPOWERED** | you can't tell yet at this sample size | collect *N* more samples (siggate computes *N*) |

It runs as a **CLI**, a **CI exit-code gate**, and a **GitHub Action that posts the verdict as a PR comment**. It is framework-neutral: adapters for
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(`--log_samples`), [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)
(`.eval` logs), and a generic CSV/JSON of per-item scores.

## The moat: it doesn't reinvent the statistics

The hard part of this is the statistics, and they are *already built and tested*.
siggate is a thin productization layer over
[**`deltagate`**](https://github.com/yongzhe2160cs/eval-reliability) — the
eval-reliability toolkit, which generalizes the paired/multiplicity/power helpers
the author contributed to Inspect AI. deltagate supplies, verbatim:

- the **paired-delta** confidence interval (the correct paired analysis on shared samples — not the unpaired test everyone reaches for),
- **Holm-Bonferroni / Benjamini-Hochberg** multiplicity correction across a suite,
- **power / minimum-sample** math (the "*N* more samples" number), and
- the **selection-bias-aware** "is this real?" probability (deflated, for best-of-N).

siggate wraps that decomposable report in a single labelled verdict, a one-line
human summary, exit codes, a PR comment, and a config file. **The stats are not
reimplemented here** — see [`gate.py`](src/siggate/gate.py).

## Install

```bash
pip install git+https://github.com/yongzhe2160cs/significance-gate
# brings in deltagate (the stats engine) automatically.
# from a clone:  pip install -e ".[dev]"
```

## CLI in sixty seconds

```bash
# Two lm-eval --log_samples files (auto-detected by extension):
siggate compare baseline.jsonl candidate.jsonl --metric acc

# Inspect logs / raw scores:
siggate compare run_a.eval run_b.eval --adapter inspect --metric match
siggate compare a.csv b.csv            # id,score columns

# A whole suite (two directories, paired by filename) with multiplicity control:
siggate compare runs_baseline/ runs_candidate/ --correction holm
```

```text
🟡 underpowered: UNDERPOWERED
   baseline 66.11%  candidate 69.44%  delta +3.33pp  (n=180)
   paired CI [-3.73pp, +10.39pp]  p=0.3547  P(real)=0.82
   min detectable @ n=180: 10.09%   -> ~1,470 more samples for an effect this size
   => 🟡 UNDERPOWERED: +3.33pp is within noise (p=0.355); you'd need ~1,470 more samples (1,650 total) to call it at 80% power.
```

The **exit code reflects the gate**, so this *is* your CI check:

```bash
siggate compare baseline.jsonl candidate.jsonl   # exit 0 = passed, 1 = blocked, 2 = error
```

By default the gate fails (exit 1) when you report an improvement that is **not**
statistically supported (`unsupported-improvement`), or on a **real regression**.
Tune it in `siggate.toml` or with `--fail-on` (see [Configuration](#configuration)).

`--format {text,json,markdown,github}` switches output; `--output comment.md`
writes the rendered PR comment to a file.

## GitHub Action quickstart

Add `.github/workflows/significance-gate.yml` (full sample
[here](.github/workflows/significance-gate.yml)):

```yaml
permissions:
  contents: read
  pull-requests: write   # so the Action can post the PR comment

jobs:
  significance-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      # ... your eval step produces baseline + candidate per-sample scores ...
      - uses: yongzhe2160cs/significance-gate@main
        with:
          baseline: eval-out/baseline.jsonl
          candidate: eval-out/candidate.jsonl
          adapter: lm-eval
          metric: acc
          fail-on: unsupported-improvement,regression,invalid
          github-token: ${{ secrets.GITHUB_TOKEN }}   # standard token; nothing hardcoded
```

On every PR the Action runs the gate, **posts (and updates) a single comment**,
writes the verdict to the job summary, and fails the check when the gate trips:

> ### 🟡 Significance Gate — UNDERPOWERED
>
> **`arc_challenge`** — +3.33pp is within noise (p=0.355); you'd need ~1,470 more samples (1,650 total) to call it at 80% power.
>
> | metric | value |
> | --- | --- |
> | delta (candidate − baseline) | **+3.33pp** |
> | baseline → candidate | 66.11% → 69.44% |
> | 95% CI | [-3.73pp, +10.39pp] |
> | p-value | 0.355 |
> | samples (n) | 180 |
> | min detectable @ n=180 | 10.09% |
>
> 🟡 **Underpowered.** This delta is below what n=180 can resolve. You'd need ~**1,470 more samples** (1,650 total) to detect an effect this size at 80% power.

It finds its prior comment by a hidden marker and edits in place, so a PR gets one
living verdict instead of a pile of bot comments. The token is the standard
`${{ secrets.GITHUB_TOKEN }}` — no secrets are baked into the Action.

## The three-way demo (the whole pitch)

```bash
python examples/make_demo_data.py   # regenerate the committed synthetic data
python examples/demo.py             # prints CLI output + PR comment for all cases
```

Bundled, realistic, lm-eval-shaped synthetic data drives three single-comparison
cases and one suite. The data is *paired* (a shared per-item difficulty makes both
models miss the same hard questions) — exactly the correlation a naive unpaired
eyeball gets wrong.

**(a) A real effect — 🟢 REAL** (`gate: PASS, exit 0`)

```text
🟢 REAL: candidate +3.67pp vs baseline (p=0.000666, 95% CI [+1.56pp, +5.78pp]) — a real change at α=0.05.
```

**(b) A noise delta — ⚪ NOISE** (`gate: FAIL, exit 1`)

```text
⚪ NOISE: +0.73pp is within noise (p=0.488); n=1,500 was enough to detect a 5.00% effect and none is there.
```

**(c) An underpowered case — 🟡 UNDERPOWERED** (`gate: FAIL, exit 1`)

```text
🟡 UNDERPOWERED: +3.33pp is within noise (p=0.355); you'd need ~1,470 more samples (1,650 total) to call it at 80% power.
```

**(d) A suite of 4 tasks — multiplicity correction in action.** Naively *two*
tasks look like wins; after Holm correction only *one* survives:

```text
suite: 4 tasks  alpha=0.05  correction=holm
   naive per-task wins : 2 ['hellaswag', 'mmlu']
   survive correction : 1 ['mmlu']
```

| task | delta | p | adj p | verdict |
|---|---|---|---|---|
| `arc_challenge` | -1.67pp | 0.091 | 0.181 | ⚪ NOISE |
| `gsm8k` | +0.42pp | 0.731 | 0.731 | ⚪ NOISE |
| `hellaswag` | +1.92pp | **0.029** | 0.086 | ⚪ NOISE |
| `mmlu` | +6.58pp | 3.2e-08 | 1.3e-07 | 🟢 REAL |

`hellaswag` clears a naive `p<0.05` — and would be reported as a win on any
dashboard — but doesn't survive correcting for four simultaneous comparisons.
That single distinction is most of what siggate is for.

## How a verdict is decided

```
REAL          significant (paired p < α, bootstrap CI excludes 0, and — if
              best-of-N — the selection-deflated probability clears 1-α);
              in a suite, also survives Holm/BH correction.
NOISE         not significant, AND the eval was powered to find a meaningful
              effect (min-detectable ≤ mei) — you looked, there's nothing there.
UNDERPOWERED  not significant, AND the eval could NOT resolve a meaningful
              effect (min-detectable > mei) — collect ~N more samples.
INVALID       a degenerate comparison (e.g. the two runs are byte-identical).
```

The NOISE-vs-UNDERPOWERED split needs an **equivalence margin** — the *minimum
effect of interest* `mei` (default `0.05`, i.e. 5 accuracy points; configurable
in your metric's units). This is the honest distinction between "confidently no
effect" and "can't tell yet", and it's the right knob for a team to set once.

## Configuration

Drop a `siggate.toml` at your repo root (every field optional, shown with defaults):

```toml
[gate]
alpha   = 0.05
power   = 0.80
mei     = 0.05   # minimum effect of interest (metric units) → NOISE vs UNDERPOWERED
fail_on = ["unsupported-improvement", "regression", "invalid"]

[metric]
name    = "acc"   # lm-eval metric key / Inspect scorer
adapter = "auto"  # auto | lm-eval | inspect | raw

[suite]
correction = "holm"   # holm (family-wise) | bh (false-discovery-rate)

[selection]
n_trials = 1      # best-of-N variants tried before reporting this run
```

`fail_on` rules (any match blocks the build): `unsupported-improvement`
(claimed a win that isn't real), `regression` (a real change in the wrong
direction), `not-real` (strict: anything not REAL), `noise`, `underpowered`,
`invalid`. CLI flags override the file; `--fail-on none` makes it report-only.

## Library use

```python
import siggate

v = siggate.compare(baseline_scores, candidate_scores, name="mmlu")
print(v.label)        # "REAL" | "NOISE" | "UNDERPOWERED" | "INVALID"
print(v.summary())    # one-line human verdict
print(v.n_more)       # samples still needed (None when already resolved)

suite = siggate.compare_suite({"mmlu": (base, cand), ...}, method="holm")
print([t.name for t in suite.survivors])   # tasks that survive correction
```

## What's MVP vs. what a hosted product needs

This open-core CLI + Action is genuinely useful today and is honest about its
edges:

**Works now:** all three adapters, paired CI + multiplicity + power + deflated
verdict (via deltagate), exit-code gate, `siggate.toml`, the PR-comment Action
(comment upsert via the standard `GITHUB_TOKEN`), suite mode, the three-way demo.

**MVP limitations (documented, not hidden):**
- `mei` defaults to `0.05` and is calibrated for accuracy-style **proportion**
  metrics; set it in your metric's units for anything else.
- Pairing requires **aligned sample ids** across runs (the adapters key on
  `doc_id` / Inspect `id` / your `id` column); siggate refuses to silently
  intersect mismatched runs.
- The power / `min-samples` math uses the normal approximation deltagate ships
  (excellent for the sample sizes evals actually use; not exact small-n).
- The Action installs from a git ref and runs the CLI; it is not yet a published
  Marketplace Action or a pinned PyPI release.

**What the hosted "Significance Gate" app would add** (the commercial layer, not
in this repo): a GitHub **App** (org-wide install, no per-repo token wiring), a
**dashboard** of eval history and power over time, **status checks** with
required-gate branch protection, trend/regression alerting, storage of eval
artifacts, and **billing**. None of that changes the math — the moat is the
statistical rigor, and that lives in the open core.

## Roadmap

- [ ] Publish `deltagate` and `siggate` to PyPI; pin a Marketplace Action.
- [ ] More adapters (OpenAI evals, HELM, raw HF `datasets` columns).
- [ ] Per-metric `mei` presets and an equivalence-test (TOST) mode for "prove no regression".
- [ ] Bootstrapped power for tiny-n / heavy-tailed metrics.
- [ ] Hosted app: GitHub App auth, dashboard, required status checks, billing.

## Related work by the same author

The eval-rigor portfolio this productizes and sits beside:

- [**deltagate** / eval-reliability](https://github.com/yongzhe2160cs/eval-reliability) — the statistics engine under siggate.
- [**agent-eval-reliability**](https://github.com/yongzhe2160cs/agent-eval-reliability) — ICC, pass@k CIs, paired tests with multiplicity for agent evals.
- [**calibration-toolkit**](https://github.com/yongzhe2160cs/calibration-toolkit) — ECE±CI, debiased estimator, Brier decomposition, temperature/Platt scaling.
- [**leaderboard-reliability**](https://github.com/yongzhe2160cs/leaderboard-reliability) — re-ranks LLM leaderboards with Wilson CIs and tie bands.

## License

MIT © 2026 yongzhe2160cs
