# LEARNS v0.0014: "Signal and Noise" (`README.md`)

**LEARNS** (*Layered Electromagnetic-Turing Autonomous Recurrent Neural System*) is a zero-dependency, single-file browser language model (`index.html`) built by ConsciousNode SoftWorks. Every component is derived from one of the studio's six theory papers, and every component can be switched on or off from the browser Switchboard or the headless Node CLI (`tests/run.js`).

> **Audit pending (Kehai, 2026-11-05):** v0.0014 is committed as Argon delivered it. The audit is scheduled after the usage reset. `results.jsonl` (233 records) wasn't in the delivery, so `tests/tables.js` can't regenerate the tables until it's added. v0.0013's results are in `versions/results-v0_0013.jsonl`.

---

## What Changed in v0.0014 ("Signal and Noise")

Following the v0.0013 audit (`AUDIT-v0.0013.md`), v0.0014 establishes a strict 5-seed paired measurement protocol with checkpoint jitter bands (`--tail 5`), replaces the unstable `fpNorm=layernorm` default with the contractive `fpNorm=rescale;fpInject=on` default, and answers why LEARNS trailed a parameter-matched GRU:

1. **Tier 1 Measurement Protocol (`--tail 5`, 5 Seeds `3, 7, 11, 13, 17`, `tests/tables.js`):**
   - `tests/run.js` now evaluates the full `56,240`-byte validation split at the final step and $K-1$ earlier checkpoints spaced 10 steps apart (`--tail 5` by default), reporting `valLoss`, `valTailMin`, `valTailMax`, `valTailMean`, and `valTailRange`.
   - `tests/tables.js` reads `results.jsonl` and generates every Markdown table in `EXPERIMENT_REPORT.md` directly from script output using paired differences across shared seeds, sample SD $s_\Delta$, two-sided 95% Student-$t$ confidence intervals ($df = n-1$), and the **Tier 1 Dual Verdict Rule**: a difference is declared **`Improves`** or **`Degrades`** only when the 95% CI of paired $\Delta$ excludes `0` **and** $|\text{mean }\Delta| > \max(\text{meanTailRange}_A, \text{meanTailRange}_B)$; otherwise it is **`Inconclusive`**.
2. **New Stable Default (`fpNorm=rescale;fpInject=on`) & Legacy Bit-Identity (`tests/repro-v0011.js`):**
   - `fpNorm=rescale;fpInject=on` is the new default (selected on 5-seed validation in Table 1), eliminating the `1.82e+9` LayerNorm-bias gradient spike of `fpNorm=layernorm` and lowering 5-seed 1,000-step test loss from `2.4416` to **`2.3408`** (`-0.1008 ± 0.0736` nats).
   - Bit-for-bit reproduction of v0.0011 and v0.0012 is preserved under the explicit legacy config `dim=32;layers=2;chunk=64;lr=0.004;fpNorm=layernorm;fpInject=off;helixState=shared;scoutScale=0.5;ema=off` with `--stream=head`, verified automatically by `node tests/repro-v0011.js`.
3. **Why LEARNS Trailed the GRU — and What Closes the Gap (Tier 3 & Tier 4):**
   - **Auxiliary Reverse-Scout Gradient Conflict (`scoutScale=0`):** Under `rbMode=causal`, `trainChunk` still ran `scoutBackwardPass` right-to-left over every chunk and injected reverse-English bigram gradients ($P(x_{t-1}\mid x_t)$) with weight `0.5` into the same shared readout matrices ($\mathbf{W}_{\text{present}}, \mathbf{W}_{\text{frontier}}$) used for forward prediction ($P(x_{t+1}\mid x_t)$). Setting `scoutScale=0` (keeping both $\mathbf{R}_A$ and $\mathbf{R}_B$ active causally with all `54,400` parameters) improves 5-seed 1,000-step ternary test loss by **`-0.1227 ± 0.0172` nats** (`2.3408 → 2.2181`) and 3-seed 7,030-step ternary test loss by **`-0.1465 ± 0.0156` nats** (`2.2999 → 2.1534`), closing within **`0.0316 ± 0.0225` nats** of the parameter-matched ternary GRU (`2.1218`, **`Inconclusive`** under the Tier 1 rule).
   - **Whole-Matrix Ternary Quantization (`weights=float32`):** Switching LEARNS to `float32` reduces checkpoint jitter (`meanTailRange`) from `0.0198` to **`0.0005` nats** (`40×` smoother) and lowers 1,000-step test loss by **`-0.0800 ± 0.0194` nats** (`2.3408 → 2.2608`). Combining `weights=float32;scoutScale=0` reaches **`2.1736`** at 1,000 steps and **`1.9941`** at 7,030 steps (beating `Transformer float32` at `2.0048` on all 3 seeds).

---

## Headline Results (`node tests/tables.js results.jsonl`)

### 1. Tier 3 & Tier 4 Architecture Comparison at 1,000 Steps (`dim=32`, 5 Seeds `3, 7, 11, 13, 17`)

### Table 3B: Tier 3 & Tier 4 Architecture & Bottleneck Comparison at `dim=32` (1,000 Steps, `--stream=stride`, 5 Seeds `3, 7, 11, 13, 17`, `--tail 5 --final`)

| Architecture / Configuration | Params | Recurrent State | Val Mean | Mean Tail Range | Test Mean | Test Seeds (`3 / 7 / 11 / 13 / 17`) | Paired $\Delta\text{Test}$ vs LEARNS | 95% Student-$t$ CI ($df=4$) | Tier 1 Verdict |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| LEARNS default (`ternary, scoutScale=0.5`) | 54,400 | 96 | 2.2960 | 0.0198 | **2.3408** | 2.3657 / 2.3325 / 2.3398 / 2.3302 / 2.3360 | — *(baseline)* | — | **Baseline** |
| LEARNS `helixState=per_layer` (`ternary`) | 54,400 | 128 | 2.8354 | 0.0050 | 2.8616 | 3.0995 / 2.5572 / 2.5351 / 3.0982 / 3.0184 | +0.5208 $\pm$ 0.2856 | `[+0.1662, +0.8754]` | **Degrades** |
| LEARNS `weights=float32` (`scoutScale=0.5`) | 54,400 | 96 | 2.2134 | 0.0005 | 2.2608 | 2.2582 / 2.2615 / 2.2805 / 2.2601 / 2.2438 | -0.0800 $\pm$ 0.0194 | `[-0.1042, -0.0559]` | **Improves** |
| LEARNS `scoutScale=0` (`ternary`, Tier 4) | 54,400 | 96 | 2.1742 | 0.0099 | 2.2181 | 2.2160 / 2.2255 / 2.2251 / 2.2006 / 2.2234 | -0.1227 $\pm$ 0.0172 | `[-0.1441, -0.1013]` | **Improves** |
| LEARNS `weights=float32;scoutScale=0` (Tier 3+4) | 54,400 | 96 | 2.1289 | 0.0007 | 2.1736 | 2.2036 / 2.1478 / 2.1702 / 2.1741 / 2.1723 | -0.1672 $\pm$ 0.0109 | `[-0.1807, -0.1537]` | **Improves** |
| Transformer `ternary` (`dim=32`) | 54,656 | KV(64) | 2.3758 | 0.0048 | 2.4126 | 2.3643 / 2.4587 / 2.3682 / 2.4106 / 2.4612 | +0.0718 $\pm$ 0.0573 | `[+0.0007, +0.1429]` | **Degrades** |
| Transformer `float32` (`dim=32`) | 54,656 | KV(64) | 2.3124 | 0.0004 | 2.3515 | 2.3495 / 2.3460 / 2.3540 / 2.3415 / 2.3663 | +0.0106 $\pm$ 0.0168 | `[-0.0102, +0.0315]` | **Inconclusive** |
| GRU `ternary` (`dim=32`) | 54,450 | 98 | 2.0664 | 0.0087 | 2.1096 | 2.1020 / 2.1246 / 2.1140 / 2.0951 / 2.1121 | -0.2313 $\pm$ 0.0206 | `[-0.2569, -0.2057]` | **Improves** |
| GRU `float32` (`dim=32`) | 54,450 | 98 | 1.9806 | 0.0005 | 2.0210 | 2.0209 / 2.0229 / 2.0143 / 2.0437 / 2.0031 | -0.3199 $\pm$ 0.0226 | `[-0.3479, -0.2918]` | **Improves** |

### 2. Full-Book Pass at 7,030 Steps (`dim=32`, 3 Seeds `3, 7, 11`)

### Table 6A: Tier 3.4 & Tier 4 Full-Book Training Pass (`7,030` Steps = 1.0 Epoch, `--stream=stride`, 3 Seeds `3, 7, 11`, `--tail 5 --final`)

| Architecture / Configuration | Val Mean | Mean Tail Range | Test Mean | Test Seeds (`3 / 7 / 11`) | Paired $\Delta\text{Test}$ vs LEARNS v0.0014 | 95% Student-$t$ CI ($df=2$) | Tier 1 Verdict |
|---|---:|---:|---:|---|---:|---|---|
| LEARNS v0.0014 default (`fpNorm=rescale;fpInject=on`) | 2.2553 | 0.0295 | **2.2999** | 2.3257 / 2.2954 / 2.2787 | — *(baseline)* | — | **Baseline** |
| LEARNS v0.0013 legacy (`fpNorm=layernorm;fpInject=off`) | 2.3366 | — | 2.3753 | 2.3497 / 2.3766 / 2.3996 | +0.0754 $\pm$ 0.0487 | `[-0.0457, +0.1964]` | **Inconclusive** |
| LEARNS `relEmbed=off` (`fpNorm=layernorm;fpInject=off`) | 2.3083 | — | 2.3472 | 2.3751 / 2.3442 / 2.3223 | +0.0473 $\pm$ 0.0032 | `[+0.0394, +0.0552]` | **Degrades** |
| LEARNS `weights=float32` (`lr=0.016`) | 2.0444 | 0.0000 | 2.0885 | 2.0852 / 2.1039 / 2.0763 | -0.2114 $\pm$ 0.0257 | `[-0.2754, -0.1475]` | **Improves** |
| LEARNS `scoutScale=0` (`ternary, lr=0.016`, Tier 4) | 2.1143 | 0.0211 | 2.1534 | 2.1657 / 2.1452 / 2.1493 | -0.1465 $\pm$ 0.0156 | `[-0.1854, -0.1077]` | **Improves** |
| LEARNS `weights=float32;scoutScale=0` (`lr=0.016`, Tier 3+4) | 1.9511 | 0.0000 | 1.9941 | 1.9968 / 1.9837 / 2.0016 | -0.3059 $\pm$ 0.0264 | `[-0.3714, -0.2403]` | **Improves** |
| Transformer `ternary` (`dim=32, lr=0.016`) | 2.1034 | — | 2.1408 | 2.1279 / 2.1822 / 2.1123 | -0.1591 $\pm$ 0.0428 | `[-0.2654, -0.0528]` | **Improves** |
| Transformer `float32` (`dim=32, lr=0.008`) | 1.9643 | — | 2.0048 | 2.0029 / 1.9960 / 2.0155 | -0.2951 $\pm$ 0.0300 | `[-0.3697, -0.2205]` | **Improves** |
| GRU `ternary` (`dim=32, lr=0.016`) | 2.0746 | 0.0565 | 2.1218 | 2.1473 / 2.1263 / 2.0917 | -0.1782 $\pm$ 0.0090 | `[-0.2004, -0.1559]` | **Improves** |
| GRU `ternary` vs LEARNS `scoutScale=0` *(paired)* | 2.0746 | 0.0565 | 2.1218 | 2.1473 / 2.1263 / 2.0917 | -0.0316 $\pm$ 0.0225 *(vs `scoutScale=0`)* | `[-0.0875, +0.0243]` | **Inconclusive** |
| GRU `float32` (`dim=32, lr=0.016`) | 1.8191 | 0.0001 | 1.8638 | 1.8739 / 1.8787 / 1.8387 | -0.4362 $\pm$ 0.0179 | `[-0.4806, -0.3917]` | **Improves** |
| GRU `float32` (`dim=32, lr=0.008`) | 1.6640 | — | 1.7044 | 1.7010 / 1.7207 / 1.6914 | -0.5955 $\pm$ 0.0261 | `[-0.6603, -0.5308]` | **Improves** |

### Table 6B: Full-Pass Validation Learning Curves (3-Seed Mean Validation Loss in nats/byte across Checkpoints `1,000 .. 7,030`)

| Architecture / Configuration | Step 1,000 | Step 2,000 | Step 3,000 | Step 4,000 | Step 5,000 | Step 6,000 | Step 7,030 (Final) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LEARNS v0.0014 default (`fpNorm=rescale;fpInject=on`) | 2.6511 | 2.6441 | 2.6161 | 2.5376 | 2.4478 | 2.3551 | **2.2553** |
| LEARNS v0.0013 legacy (`fpNorm=layernorm;fpInject=off`) | 2.6227 | 2.7691 | 2.6034 | 2.6448 | 2.5130 | 2.4463 | **2.3366** |
| LEARNS `relEmbed=off` (`fpNorm=layernorm;fpInject=off`) | 2.6941 | 2.7329 | 2.6411 | 2.5355 | 2.4824 | 2.4388 | **2.3083** |
| LEARNS `weights=float32` (`lr=0.016`) | 2.4726 | 2.4219 | 2.3335 | 2.2196 | 2.1233 | 2.0606 | **2.0444** |
| LEARNS `scoutScale=0` (`ternary, lr=0.016`, Tier 4) | 2.5740 | 2.5240 | 2.4912 | 2.4290 | 2.3261 | 2.2139 | **2.1143** |
| LEARNS `weights=float32;scoutScale=0` (`lr=0.016`, Tier 3+4) | 2.4383 | 2.3357 | 2.2462 | 2.1376 | 2.0415 | 1.9732 | **1.9511** |
| Transformer `ternary` (`dim=32, lr=0.016`) | 2.4911 | 2.3985 | 2.3417 | 2.2623 | 2.2016 | 2.1366 | **2.1034** |
| Transformer `float32` (`dim=32, lr=0.008`) | 2.3658 | 2.2616 | 2.1617 | 2.0880 | 2.0258 | 1.9823 | **1.9643** |
| GRU `ternary` (`dim=32, lr=0.016`) | 2.3556 | 2.4285 | 2.4423 | 2.3331 | 2.2636 | 2.2211 | **2.0746** |
| GRU `float32` (`dim=32, lr=0.016`) | 2.2031 | 2.1289 | 2.0340 | 1.9553 | 1.8726 | 1.8298 | **1.8191** |
| GRU `float32` (`dim=32, lr=0.008`) | 2.0198 | 1.9203 | 1.8369 | 1.7563 | 1.7064 | 1.6668 | **1.6640** |

---

## Reproducing Everything

```bash
# 1. Verify bit-identical v0.0011/v0.0012 reproduction under explicit legacy config
node tests/repro-v0011.js

# 2. Run analytical vs finite-difference gradient check across all 41 configurations
node tests/gradcheck.js --all

# 3. Verify optimizer parameter coverage across all 42 configurations
node tests/optimizer-coverage.js --all

# 4. Run a single 1,000-step experiment with 5-checkpoint validation tail band (--tail 5)
node tests/run.js --arch learns --config "dim=32;lr=0.016;scoutScale=0" --seed 3 --steps 1000 --stream stride --tail 5 --final

# 5. Recompute every table in EXPERIMENT_REPORT.md directly from results.jsonl
node tests/tables.js results.jsonl
```
