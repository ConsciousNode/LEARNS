# AUDIT: LEARNS v0.0013 "The Long Run"

**Audited:** Argon's delivery (commit `8ec6143`) · **Auditor:** Kehai Interim · **Date:** 2026-11-04

**Verdict:** the numbers are real, and every one I checked reproduces bit for bit. Gradients, optimizer coverage, the
v0.0012 bit-identity claim and the contractivity proof all hold. Three of the report's *interpretations* go further
than the data: one explanation is factually wrong, the headline fixed-point gain is mostly one bad baseline seed, and
several single-seed effects are smaller than the run-to-run wobble of a ternary model. No code changes were needed.

## What reproduces

| claim | check | result |
|---|---|---|
| `CODE_HASH 0xdf8ae499`, `CORPUS_HASH 0x39c84ca9` | `tests/load.js` | match |
| Default switches + `--stream=head` = v0.0012 bit for bit | 150 steps, seed 3, v0.0013 vs `versions/learns-v0_0012.html` | identical train/val/test to 8 decimals (test `3.156262`) |
| Gradients, all switches and baselines | `node tests/gradcheck.js --all` | **36/36 ALL OK** |
| Optimizer applies every gradient | `node tests/optimizer-coverage.js --all` | **37/37 ALL OK** |
| Full pass, LEARNS default, seed 3 | `--steps 7030 --curve 500 --final` | val `2.310062`, test `2.349672`, **and the whole val curve**: identical to `results.jsonl` |
| Full pass, GRU float32, seed 3 | same | test `1.70098`: identical |
| "Best LEARNS result" (`fpNorm=none;fpInject=on`, 3,515 steps) | same | val `2.236392`, test `2.275791`: identical |
| LR sweep points | GRU ternary and LEARNS at `lr=0.016`, 1,000 steps | `2.058522` and `2.316919`: identical |
| Every table in README / EXPERIMENT_REPORT | recomputed from the 140-run `results.jsonl` | all means and per-seed values match |
| `--curve` evaluations don't disturb training | GRU 600 steps and LEARNS 300 steps, curve on vs off | identical to 8 decimals |
| `W._qcache` speed-up never goes stale | read the code | invalidated at the start and end of `optimizerStep`, after the spectral clip and gamma clamp. Sound |
| Contractivity (`fpNorm=none\|rescale`): Lip ≤ 0.9 | the loop uses float `matvec` on W_FA/W_FB (the clipped matrices, not ternary copies); exact SVD after 300 steps | σ_max = 0.779, 0.870, **0.900**, 0.801. The bound holds, and the `rescale` map is non-expanding as claimed |

## Where the report goes further than its data

### 1. "The first 9,600 bytes are the Project Gutenberg header": they aren't
`corpus.txt` has no Gutenberg header (`grep -c Gutenberg` = 0). The title and contents take **579 bytes**, and the
narrative starts at byte 579. So 94% of the bytes the `head` stream sees at 150 steps are ordinary story text. The
measured head→stride gain is real (and reproduces), but the stated reason is wrong. The likelier one: `head` trains
150 times on one contiguous stretch of one story, while `stride` samples 150 chunks from across the whole book. Why
LEARNS gains 2–3× more than GRU/Transformer is still an open question, not an answered one.

### 2. Ternary models wobble by ~0.01–0.016 nats from one step to the next, even at the LR floor
`tests/jitter.js` (new) trains exactly as `run.js` does and scores the full validation split at several of the final
steps. By then the cosine schedule is at its floor (`lrMin = 1e-5`):

| model (seed 3, 1,000 steps) | val loss at steps 960 / 970 / 980 / 990 / 1000 | range |
|---|---|---:|
| LEARNS default (ternary) | 2.3187 / 2.3251 / 2.3131 / 2.3092 / 2.3169 | **0.016** |
| GRU ternary | 2.0591 / 2.0641 / 2.0540 / 2.0533 / 2.0585 | **0.011** |
| GRU float32 | 1.9573 / 1.9571 / 1.9570 / 1.9569 / 1.9569 | 0.0004 |

Ternary weights round to −1/0/+1 against a scale set by the mean |W|, so updates of ~1e-5 still flip roundings. This
also explains the jump in the report's own curve, from 2.2794 at step 7,000 to 2.3101 at step 7,030. It isn't a bug:
it's how ternary models behave. **Consequence:** a single-seed, single-checkpoint difference under about 0.03 nats
can't be told apart from where the run happened to stop. That covers these claims:
- "`relEmbed=on` beats `relEmbed=off` by +0.025 on seed 3 at 7,030 steps";
- "helps on seed 3" for coreA (+0.030), coreB (+0.023), thermo (+0.033) and helixRetention (+0.040);
- the half-pass deltas for `rescale` (−0.0048) and `none` (−0.0084).

The report already marks most of these Inconclusive in its verdict column. The README's component table states them
as "Helps on seed=3", which it shouldn't. The half-pass `none;fpInject=on` gain (−0.0394, single seed) is about 2.5×
the jitter: suggestive, not established.

### 3. The fixed-point "−0.105 nats" is mostly one bad baseline seed
At 1,000 steps the default-LEARNS baseline scores **2.5381 on seed 7**, about 0.15 nats worse than its seeds 3 and 11
(2.3653, 2.4494). Every ablation is paired against that baseline, so almost every switch "improves" seed 7 by 0.13–0.21:

| vs default LEARNS (test, 1,000 steps) | Δ seed 3 | Δ seed 7 | Δ seed 11 |
|---|---:|---:|---:|
| `fpNorm=rescale;fpInject=on` (headline −0.1049) | +0.0004 | −0.2056 | −0.1096 |
| `fpNorm=none;fpInject=on` (−0.0944) | −0.0116 | −0.2012 | −0.0703 |
| `fpNorm=none` | −0.0294 | −0.1674 | −0.0517 |
| `fpNorm=rescale` | +0.0374 | −0.1469 | −0.1234 |
| `fpIters=0` | −0.0247 | −0.1964 | −0.0978 |
| `coreB=off` | +0.0232 | −0.1979 | −0.1089 |

The honest reading: under `fpNorm=layernorm` the fixed point is **unstable**. The report itself documents 4.2e8
LayerNorm-bias gradient spikes, and seed 7 is what that instability costs. The contractive modes remove that failure,
which is a real and worthwhile result. As an *accuracy* effect on seeds where layernorm didn't blow up, it's mixed: on
seed 3 it's inside the jitter, and on seed 11 it's −0.05 to −0.12. So the report's "a fixed point that actually
converges helps" should read "a fixed point that converges removes a training instability; any accuracy gain beyond
that needs more seeds". The same seed-7 effect makes "fpIters=0 is −0.106 better" and the coreA/coreB/thermo "mixed
across seeds" rows look bigger than they are.

**For v0.0014:** compare ablations against a baseline that doesn't blow up (e.g. `fpNorm=rescale` as the default),
use 5+ seeds, and report seed-paired differences with their spread.

### 4. Smaller things
- "30–40% of LEARNS's remaining gap is caused by the non-convergent LayerNorm" is an estimate, not a measurement.
  Treat it as a hypothesis.
- The 7,030-step leaderboard and the "LEARNS plateaus while GRU/Transformer keep improving" analysis stand. LEARNS
  ternary trails the ternary GRU (2.1218) and ternary Transformer (2.1408) by about 0.2 nats at equal budget and
  parameter count, and every neural model beats the bigram. The Kneser–Ney 5-gram still leads everything.

## Files added by this audit
- `tests/jitter.js`: end-of-run checkpoint jitter, mirroring `run.js`.
- `results-audit-jitter.jsonl`: the three runs above.
- This file. README and EXPERIMENT_REPORT carry a short banner pointing here.
