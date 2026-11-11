# LEARNS v0.0015: "The Partner Record" (`README.md`)

> **Audited 2026-11-11 (Kehai Interim).** Clean. The leak test's legacy positive control fails as it must. gradcheck passes 45/45, optimizer coverage 46/46, and legacy reproduction is bit for bit. Every table regenerates verbatim from `results.jsonl`. Reused runs keep honest provenance. Independent reruns of lookback=16 (seed 3) and B0 (seed 7) match bit for bit. See `AUDIT-v0.0015.md`.

**LEARNS** (*Logical Entropic Arrow Recursive Neuro-Symbolic Architecture*) is a zero-dependency, single-file browser language model (`index.html`) built by ConsciousNode SoftWorks. Every component is derived from one of the studio's six theory papers, and every component can be switched on or off from the browser Switchboard or the headless Node CLI (`tests/run.js`).

---

## What Changed in v0.0015 ("The Partner Record")

Following the v0.0014 audit (`AUDIT-v0.0014.md`), v0.0015 promotes `scoutScale=0` to the default, adds an automated causality leak and lookback-equivalence test suite (`tests/leak.js`), enforces strict provenance for reused runs (`reusedFrom` + original `codeHash`, marked with `†` by `tests/tables.js`) and strict `Measured:` vs. `Hypothesis:` reporting, and investigates Component 4 (*It's Time To Toss The Dice*: the partner record $\mathbf{R}_B$):

1. **Tier 1 — New Default (`scoutScale=0`) and Automated Causality Leak Suite (`tests/leak.js`):**
   - **Measured:** `scoutScale=0` is now the v0.0015 default (**B0**: `scoutScale=0;rbMode=causal;fpNorm=rescale;fpInject=on`). Legacy configs (`v0.0011`, `v0.0012`, and `v0.0014` with `scoutScale=0.5`) reproduce bit-for-bit via `node tests/repro-v0011.js`.
   - **Measured:** `node tests/leak.js` verifies on both random weights and a trained checkpoint that perturbing any byte after position $t$ leaves every logit at or before $t$ bit-identical (`max |Δlogit| === 0`) across `rbMode=causal`, `recordB=off`, `rbMode=prevrev`, `rbMode=lookback;lookback=16`, `rbMode=lookback;lookback=64`, and `scoutScale=0.5;scoutHead=separate`, while `rbMode=legacy` fails as the positive control (`max |Δlogit| = 1.087e-1` / `3.864e-2`). It also proves that `rbMode=lookback` in `forward()` and `_forwardEval()` is bit-identical (`max |ΔR_B| === 0`) to a standalone scratch window recurrence.
2. **Tier 2 — What Should the Partner Record $\mathbf{R}_B$ Be? (1,000 Steps, 5 Seeds `3, 7, 11, 13, 17`):**
   - **Measured (Q1 — Does $\mathbf{R}_B$ earn its parameters when the reverse loss is off? A1 vs. B0):** **No (`Inconclusive`).** Disabling $\mathbf{R}_B$ (**A1**: `recordB=off`, saving 3,072 parameters `54,400 -> 51,328` and speeding up training from `1,305` to `1,398` tok/s) gives test loss `2.2194` vs. `2.2181` for **B0** ($\Delta\text{Test} = +0.0013 \pm 0.0256$, 95% CI `[-0.0305, +0.0330]`, **Inconclusive**). In **B0**, $\mathbf{R}_B$ maintains norm $\|\mathbf{R}_B\|_2 = 5.1572$ but converges to an anti-correlated vector ($\cos(\mathbf{R}_A, \mathbf{R}_B) = -0.7744$).
   - **Measured (Q2 — Does a backward read of the recent past beat a forward carry? A2 and A3 vs. B0):** **No (`Inconclusive`).** Neither `rbMode=prevrev` (**A2**: `2.2181`, $\Delta\text{Test} = +0.0000$, **Inconclusive**, `706` tok/s) nor Kham's rolling backward window `rbMode=lookback` (**A3a** `lookback=16`: `2.2188`, $\Delta\text{Test} = +0.0007 \pm 0.0177$, **Inconclusive**, `927` tok/s; **A3b** `lookback=64`: `2.2228`, $\Delta\text{Test} = +0.0046 \pm 0.0111$, **Inconclusive**, `657` tok/s) improves on forward carry **B0** (`2.2181`, `1,305` tok/s).
   - **Measured (Q3 — Did `scoutScale=0.5` hurt because of the shared readout, or because reverse-LM gradients hurt the core itself? A4 vs. B0 vs. v0.0014 default):** **Both — 86.2% from the shared readout and 13.8% from reverse-LM gradients into the core.** Giving the reverse-LM scout its own readout head (**A4**: `scoutScale=0.5;scoutHead=separate`) improves test loss vs. shared-head `scoutScale=0.5` (`2.3408 †`) by **$-0.1057 \pm 0.0149$ nats** (`2.2351`, 95% CI `[-0.1241, -0.0873]`, **Improves**), recovering 86.2% of the `+0.1227`-nat penalty. However, **A4** (`2.2351`) still trails **B0** (`scoutScale=0`, `2.2181`) on all 5 seeds by **$+0.0170 \pm 0.0066$ nats** (95% CI `[+0.0088, +0.0251]`, **Degrades**).
3. **Tier 3 — Full-Book Pass (`7,030` Steps, 3 Seeds `3, 7, 11`, Fresh v0.0015 Code `0x626fd26c`):**
   - **Measured:** Because no Tier 2 arm improved on **B0** at 1,000 steps, **B0** and **`GRU ternary`** were both run fresh for `7,030` steps (`--tail 5 --final`) under v0.0015 code (`0x626fd26c`). At `7,030` steps, **B0** reaches `2.1534` (`meanTailRange = 0.0211`) and **`GRU ternary`** reaches `2.1218` (`meanTailRange = 0.0565`), a paired difference of $\Delta\text{Test} = -0.0316 \pm 0.0225$ nats (95% CI `[-0.0875, +0.0243]`, **Inconclusive** at $n=3$ due to sample size and ternary GRU tail jitter; at 1,000 steps across 5 seeds, `GRU ternary` beats **B0** by $-0.1086 \pm 0.0053$ nats, **Improves**).

---

## Headline Results (`node tests/tables.js results.jsonl`)

### Table 1A: Tier 2 Partner Record Study — Held-Out Test Loss & Verdicts (1,000 Steps, `--stream=stride`, 5 Seeds `3, 7, 11, 13, 17`, `--tail 5 --final`)

| Arm / Configuration | Partner Record Mechanism | Val Mean | Mean Tail Range | Test Mean | Test Seeds (`3 / 7 / 11 / 13 / 17`) | Paired $\Delta\text{Test}$ Mean $\pm s_\Delta$ | 95% Student-$t$ CI ($df=4$) | Max Tail Range | Tier 1 Verdict |
|---|---|---:|---:|---:|---|---:|---|---:|---|
| **B0**: `scoutScale=0;rbMode=causal` *(v0.0015 default)* | Forward causal carry $R_B$, no reverse-LM loss | 2.1742 | 0.0099 | **2.2181** | 2.2160 / 2.2255 / 2.2251 / 2.2006 / 2.2234 | — *(baseline B0)* | — | 0.0099 | **Baseline** |
| **A1**: `recordB=off` (`scoutScale=0`) | Disable $R_B$ ($W_{\text{rec},B}, W_{\text{cross},A}, W_{\text{cross},B}$ off) | 2.1769 | 0.0077 | 2.2194 | 2.1989 / 2.2293 / 2.2124 / 2.2454 / 2.2111 | +0.0013 $\pm$ 0.0256 | `[-0.0305, +0.0330]` | 0.0099 | **Inconclusive** |
| **A2**: `rbMode=prevrev` (`scoutScale=0`) | Seed $R_B$ at chunk start from reversed prev chunk | 2.1742 | 0.0099 | 2.2181 | 2.2160 / 2.2255 / 2.2251 / 2.2006 / 2.2234 | +0.0000 $\pm$ 0.0000 | `[-0.0000, +0.0000]` | 0.0099 | **Inconclusive** |
| **A3a**: `rbMode=lookback;lookback=16` (`scoutScale=0`) | Rolling backward recurrence over $x_t \dots x_{t-15}$ | 2.1766 | 0.0098 | 2.2188 | 2.2247 / 2.1972 / 2.2222 / 2.2177 / 2.2324 | +0.0007 $\pm$ 0.0177 | `[-0.0213, +0.0227]` | 0.0099 | **Inconclusive** |
| **A3b**: `rbMode=lookback;lookback=64` (`scoutScale=0`) | Rolling backward recurrence over $x_t \dots x_{t-63}$ | 2.1796 | 0.0111 | 2.2228 | 2.2328 / 2.2216 / 2.2162 / 2.2061 / 2.2372 | +0.0046 $\pm$ 0.0111 | `[-0.0091, +0.0184]` | 0.0111 | **Inconclusive** |
| **A4**: `scoutScale=0.5;scoutHead=separate` | Reverse-LM scout on dedicated $W_{\text{frontier,scout}}, W_{\text{present,scout}}$ | 2.1945 | 0.0115 | 2.2351 | 2.2401 / 2.2348 / 2.2394 / 2.2142 / 2.2471 | +0.0170 $\pm$ 0.0066 | `[+0.0088, +0.0251]` | 0.0115 | **Degrades** |
| v0.0014 default (`scoutScale=0.5;scoutHead=shared`) † | Reverse-LM scout sharing forward readout $W_{\text{frontier}}, W_{\text{present}}$ | 2.2960 | 0.0198 | 2.3408 | 2.3657 / 2.3325 / 2.3398 / 2.3302 / 2.3360 | +0.1227 $\pm$ 0.0172 | `[+0.1013, +0.1441]` | 0.0198 | **Degrades** |
| **A4** vs v0.0014 default † *(paired readout isolation)* | Effect of separating scout readout head at `scoutScale=0.5` | 2.1945 | 0.0115 | 2.2351 | 2.2401 / 2.2348 / 2.2394 / 2.2142 / 2.2471 | -0.1057 $\pm$ 0.0149 *(vs shared scout †)* | `[-0.1241, -0.0873]` | 0.0198 | **Improves** |
| GRU `ternary` (`dim=32, lr=0.016`) † | Parameter-matched 2-layer ternary GRU baseline | 2.0664 | 0.0087 | 2.1096 | 2.1020 / 2.1246 / 2.1140 / 2.0951 / 2.1121 | -0.1086 $\pm$ 0.0053 | `[-0.1152, -0.1020]` | 0.0099 | **Improves** |

*Note: Rows marked with `†` reuse bit-identical runs from `results.jsonl` with their original `codeHash` (`0x4048aacb` for v0.0014) and `reusedFrom` provenance field preserved.*

### Table 2A: Tier 3 Full-Book Training Pass (`7,030` Steps = 1.0 Epoch, `--stream=stride`, 3 Seeds `3, 7, 11`, `--tail 5 --final`)

| Architecture / Configuration | `codeHash` | Val Mean | Mean Tail Range | Test Mean | Test Seeds (`3 / 7 / 11`) | Paired $\Delta\text{Test}$ vs B0 | 95% Student-$t$ CI ($df=2$) | Max Tail Range | Tier 1 Verdict |
|---|---|---:|---:|---:|---|---:|---|---:|---|
| LEARNS v0.0015 default (**B0**: `scoutScale=0, rbMode=causal`) | `0x626fd26c` *(fresh v0.0015)* | 2.1143 | 0.0211 | **2.1534** | 2.1657 / 2.1452 / 2.1493 | — *(baseline B0)* | — | 0.0211 | **Baseline** |
| GRU `ternary` (`dim=32, lr=0.016`) | `0x626fd26c` *(fresh v0.0015)* | 2.0746 | 0.0565 | 2.1218 | 2.1473 / 2.1263 / 2.0917 | -0.0316 $\pm$ 0.0225 | `[-0.0875, +0.0243]` | 0.0565 | **Inconclusive** |
| LEARNS v0.0014 default (`scoutScale=0.5, rescale, inject`) † | `0x4048aacb` *(v0.0014)* | 2.2553 | 0.0295 | 2.2999 | 2.3257 / 2.2954 / 2.2787 | +0.1465 $\pm$ 0.0156 | `[+0.1077, +0.1854]` | 0.0295 | **Degrades** |
| LEARNS `weights=float32;scoutScale=0` (`lr=0.016`) † | `0x4048aacb` *(v0.0014)* | 1.9511 | 0.0000 | 1.9941 | 1.9968 / 1.9837 / 2.0016 | -0.1593 $\pm$ 0.0107 | `[-0.1860, -0.1326]` | 0.0211 | **Improves** |
| LEARNS `weights=float32;scoutScale=0.5` (`lr=0.016`) † | `0x4048aacb` *(v0.0014)* | 2.0444 | 0.0000 | 2.0885 | 2.0852 / 2.1039 / 2.0763 | -0.0649 $\pm$ 0.0208 | `[-0.1166, -0.0132]` | 0.0211 | **Improves** |
| LEARNS v0.0013 legacy (`layernorm, no-inject, scoutScale=0.5`) † | `0xdf8ae499` *(v0.0013)* | 2.3366 | — | 2.3753 | 2.3497 / 2.3766 / 2.3996 | +0.2219 $\pm$ 0.0342 | `[+0.1371, +0.3067]` | 0.0211 | **Degrades** |
| Transformer `ternary` (`dim=32, lr=0.016`) † | `0xdf8ae499` *(v0.0013)* | 2.1034 | — | 2.1408 | 2.1279 / 2.1822 / 2.1123 | -0.0126 $\pm$ 0.0429 | `[-0.1192, +0.0941]` | 0.0211 | **Inconclusive** |
| Transformer `float32` (`dim=32, lr=0.008`) † | `0xdf8ae499` *(v0.0013)* | 1.9643 | — | 2.0048 | 2.0029 / 1.9960 / 2.0155 | -0.1486 $\pm$ 0.0145 | `[-0.1846, -0.1126]` | 0.0211 | **Improves** |
| GRU `float32` (`dim=32, lr=0.016`) † | `0x4048aacb` *(v0.0014)* | 1.8191 | 0.0001 | 1.8638 | 1.8739 / 1.8787 / 1.8387 | -0.2896 $\pm$ 0.0221 | `[-0.3446, -0.2347]` | 0.0211 | **Improves** |
| GRU `float32` (`dim=32, lr=0.008`) † | `0xdf8ae499` *(v0.0013)* | 1.6640 | — | 1.7044 | 1.7010 / 1.7207 / 1.6914 | -0.4490 $\pm$ 0.0215 | `[-0.5025, -0.3955]` | 0.0211 | **Improves** |

*Note: The first two rows (`LEARNS v0.0015 default B0` and `GRU ternary`) are fresh v0.0015 runs (`codeHash: 0x626fd26c`) with full 5-checkpoint tail bands (`--tail 5 --final`). Context rows marked with `†` reuse prior-release runs with their original `codeHash` (`0x4048aacb` or `0xdf8ae499`) and `reusedFrom` field preserved.*

### Table 2B: Full-Pass Validation Learning Curves (3-Seed Mean Validation Loss in nats/byte across Checkpoints `1,000 .. 7,030`)

| Architecture / Configuration | `codeHash` | Step 1,000 | Step 2,000 | Step 3,000 | Step 4,000 | Step 5,000 | Step 6,000 | Step 7,030 (Final) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LEARNS v0.0015 default (**B0**: `scoutScale=0, rbMode=causal`) | `0x626fd26c` *(fresh v0.0015)* | 2.5740 | 2.5240 | 2.4912 | 2.4290 | 2.3261 | 2.2139 | **2.1143** |
| GRU `ternary` (`dim=32, lr=0.016`) | `0x626fd26c` *(fresh v0.0015)* | 2.3556 | 2.4285 | 2.4423 | 2.3331 | 2.2636 | 2.2211 | **2.0746** |
| LEARNS v0.0014 default (`scoutScale=0.5, rescale, inject`) † | `0x4048aacb` *(v0.0014)* | 2.6511 | 2.6441 | 2.6161 | 2.5376 | 2.4478 | 2.3551 | **2.2553** |
| LEARNS `weights=float32;scoutScale=0` (`lr=0.016`) † | `0x4048aacb` *(v0.0014)* | 2.4383 | 2.3357 | 2.2462 | 2.1376 | 2.0415 | 1.9732 | **1.9511** |
| LEARNS `weights=float32;scoutScale=0.5` (`lr=0.016`) † | `0x4048aacb` *(v0.0014)* | 2.4726 | 2.4219 | 2.3335 | 2.2196 | 2.1233 | 2.0606 | **2.0444** |
| LEARNS v0.0013 legacy (`layernorm, no-inject, scoutScale=0.5`) † | `0xdf8ae499` *(v0.0013)* | 2.6227 | 2.7691 | 2.6034 | 2.6448 | 2.5130 | 2.4463 | **2.3366** |
| Transformer `ternary` (`dim=32, lr=0.016`) † | `0xdf8ae499` *(v0.0013)* | 2.4911 | 2.3985 | 2.3417 | 2.2623 | 2.2016 | 2.1366 | **2.1034** |
| Transformer `float32` (`dim=32, lr=0.008`) † | `0xdf8ae499` *(v0.0013)* | 2.3658 | 2.2616 | 2.1617 | 2.0880 | 2.0258 | 1.9823 | **1.9643** |
| GRU `float32` (`dim=32, lr=0.016`) † | `0x4048aacb` *(v0.0014)* | 2.2031 | 2.1289 | 2.0340 | 1.9553 | 1.8726 | 1.8298 | **1.8191** |
| GRU `float32` (`dim=32, lr=0.008`) † | `0xdf8ae499` *(v0.0013)* | 2.0198 | 1.9203 | 1.8369 | 1.7563 | 1.7064 | 1.6668 | **1.6640** |

---

## Reproducing Everything

```bash
# 1. Run causality leak & lookback window scratch-equivalence suite (random + trained weights)
node tests/leak.js

# 2. Verify bit-identical v0.0011/v0.0012 (--stream=head) and v0.0014 (1,000-step scoutScale=0.5) reproduction
node tests/repro-v0011.js

# 3. Run analytical vs finite-difference gradient check across all 45 configurations
node tests/gradcheck.js --all

# 4. Verify optimizer parameter coverage across all 46 configurations
node tests/optimizer-coverage.js --all

# 5. Run a single 1,000-step experiment with 5-checkpoint validation tail band (--tail 5)
node tests/run.js --arch learns --config "dim=32;lr=0.016" --seed 3 --steps 1000 --stream stride --tail 5 --final

# 6. Recompute every table in EXPERIMENT_REPORT.md directly from results.jsonl
node tests/tables.js results.jsonl
```
