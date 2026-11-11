# LEARNS v0.0015 — Experiment Report: *The Partner Record*

- **Release:** LEARNS `v0.0015` ("The Partner Record")
- **Code Hash (`CODE_HASH`):** `0x626fd26c` (FNV-1a 32-bit over `/*@@LEARNS_BEGIN*/ ... /*@@LEARNS_END*/`; reused records preserve their original `codeHash`: `0x4048aacb` for v0.0014 and `0xdf8ae499` for v0.0013, plus `reusedFrom` provenance, marked with `†`)
- **Corpus Hash (`CORPUS_HASH`):** `0x39c84ca9` (*The Adventures of Sherlock Holmes*, 562,404 ASCII bytes; deterministic contiguous 80 / 10 / 10 split: `449,923` train / `56,240` validation / `56,241` test bytes)
- **Measurement Protocol (Tier 1, Unchanged from v0.0014):**
  - **Five seeds (`3, 7, 11, 13, 17`)** for every 1,000-step study; **three seeds (`3, 7, 11`)** for full-pass (`7,030`-step) runs.
  - **Checkpoint jitter band (`--tail 5`):** every run evaluates the full 56,240-byte validation split at the final step and at the 4 preceding checkpoints spaced 10 steps apart (`steps - 40, steps - 30, steps - 20, steps - 10, steps`), recording `valTail`, `valTailMin`, `valTailMax`, `valTailMean`, and `valTailRange = valTailMax - valTailMin`. Held-out test (`56,241` bytes) is evaluated once at the final step (`--final`).
  - **Tier 1 Dual Verdict Rule (`tests/tables.js`):** a configuration is labeled **Improves** or **Degrades** relative to baseline $B$ if and only if **(a)** the two-sided 95% Student-$t$ confidence interval of the paired per-seed differences $\Delta_i = L_{\text{var}}(s_i) - L_{\text{base}}(s_i)$ ($\bar{\Delta} \pm t_{0.975,\,n-1}\cdot s_\Delta/\sqrt{n}$, with $t_{0.975,4}=2.7764$ for $n=5$ and $t_{0.975,2}=4.3027$ for $n=3$) excludes zero **AND** **(b)** $|\bar{\Delta}| > \max(\overline{\text{valTailRange}}_{\text{base}},\,\overline{\text{valTailRange}}_{\text{var}})$. Otherwise the verdict is **Inconclusive**.
  - **Standing Audit Rules (from `AUDIT-v0.0014.md`):**
    1. **Reused-run provenance:** every record reused from v0.0013 or v0.0014 keeps its original `codeHash` (`0xdf8ae499` or `0x4048aacb`) and carries a `reusedFrom` field; `tests/tables.js` automatically marks any row that uses reused records with `†`.
    2. **Measured vs. Explained:** empirical observations are labeled **`Measured:`** and mechanistic interpretations not directly isolated by an experiment are labeled **`Hypothesis:`**. `"Inconclusive"` means the statistical test cannot distinguish the configurations at the given sample size and jitter level; it never means `"equivalent"`.

---

## 1. Tier 1: New Default (`scoutScale=0`) and Automated Causality Leak Suite (`tests/leak.js`)

### 1.1 Promotion of `scoutScale=0` to the v0.0015 Default (`B0`)
- **Measured:** In v0.0014 Tier 4, setting `scoutScale=0` (disabling the reverse-chunk auxiliary language-modeling loss while keeping causal $\mathbf{R}_B$ active) improved 1,000-step 5-seed held-out test loss from `2.3408` to `2.2181` nats/byte ($\Delta\text{Test} = -0.1227 \pm 0.0172$, 95% CI `[-0.1441, -0.1013]`, **Improves**) and 7,030-step 3-seed test loss from `2.2999` to `2.1534` nats/byte ($\Delta\text{Test} = -0.1465 \pm 0.0156$, 95% CI `[-0.1854, -0.1077]`, **Improves**). In v0.0015 (`CODE_HASH = 0x626fd26c`), `scoutScale=0` is promoted to `DEFAULT_SWITCHES.scoutScale = 0`.
- **Measured (Bit-Exact Legacy Reproduction):** Running `node tests/repro-v0011.js` verifies three backward-compatibility guarantees:
  1. Explicit v0.0011/v0.0012 config (`dim=32;layers=2;chunk=64;lr=0.004;fpNorm=layernorm;fpInject=off;helixState=shared;scoutScale=0.5;ema=off`, `--stream=head`, seed 3) reproduces v0.0011's 40-step training and thermodynamic losses to all 12 printed decimal places (`step 1: 5.544955097855`, `step 40: 3.509413177366`).
  2. The 150-step `--stream=head` run reproduces v0.0012 bit-for-bit (`valLoss = 3.137877`, `testLoss = 3.152259`).
  3. The 1,000-step v0.0014 default config (`dim=32;layers=2;chunk=64;lr=0.016;fpNorm=rescale;fpInject=on;scoutScale=0.5`, `--stream=stride`, seed 3, `--tail 5 --final`) reproduces its v0.0014 record in `results.jsonl` bit-for-bit (`trainLossFirst = 5.54189737`, `trainLossLast = 2.15520647`, `trainLoss = 2.299218`, `valLoss = 2.321951`, `valTailMin = 2.319591`, `valTailMax = 2.339794`, `valTailMean = 2.326029`, `testLoss = 2.365709`).

### 1.2 Automated Causality Leak Test & Lookback Equivalence (`tests/leak.js`)
- **Measured:** `node tests/leak.js` tests every partner-record configuration (`rbMode=causal`, `recordB=off`, `rbMode=prevrev`, `rbMode=lookback;lookback=16`, `rbMode=lookback;lookback=64`, `scoutScale=0.5;scoutHead=separate`, and `rbMode=legacy`) on **both random weights (`seed=7`) and a 15-step trained checkpoint**, across both `forward()` (`evalOnly: false`) and `_forwardEval()` (`evalOnly: true`). Perturbing future tokens at positions $p > t$ (`t + 1`, `t + 4`, `len - 1`, `len`) across probe steps $t \in \{0, 3, 7, 15, 23, 30\}$ yields:
  - `rbMode=causal` (**B0**): `random max |Δlogit| = 0.000e+0`, `trained max |Δlogit| = 0.000e+0` (**PASS**)
  - `recordB=off` (**A1**): `random max |Δlogit| = 0.000e+0`, `trained max |Δlogit| = 0.000e+0` (**PASS**)
  - `rbMode=prevrev` (**A2**): `random max |Δlogit| = 0.000e+0`, `trained max |Δlogit| = 0.000e+0` (**PASS**)
  - `rbMode=lookback;lookback=16` (**A3a**): `random max |Δlogit| = 0.000e+0`, `trained max |Δlogit| = 0.000e+0`, scratch-window equivalence `max |ΔR_B| = 0.00e+0` (**PASS**)
  - `rbMode=lookback;lookback=64` (**A3b**): `random max |Δlogit| = 0.000e+0`, `trained max |Δlogit| = 0.000e+0`, scratch-window equivalence `max |ΔR_B| = 0.00e+0` (**PASS**)
  - `scoutScale=0.5;scoutHead=separate` (**A4**): `random max |Δlogit| = 0.000e+0`, `trained max |Δlogit| = 0.000e+0` (**PASS**)
  - `rbMode=legacy` (**Positive Control**): `random max |Δlogit| = 1.087e-1`, `trained max |Δlogit| = 3.864e-2` (**LEAK DETECTED — Positive Control OK**)
- **Measured (Gradient & Optimizer Verification):** `node tests/gradcheck.js --all` passes **45/45** configurations (`cos > 0.9999`, `|ratio - 1| < 1e-3`, including `rbMode=lookback;lookback=16`, `rbMode=lookback;lookback=64`, and end-to-end forward + scout finite-difference verification of `scoutScale=0.5;scoutHead=separate` including `W_frontier_scout` and `W_present_scout`), and `node tests/optimizer-coverage.js --all` passes **46/46** configurations (`30/30` active parameter tensors stepped on `scoutScale=0.5;scoutHead=separate`).

---

## 2. Tier 2: What Should the Partner Record $\mathbf{R}_B$ Be? (1,000 Steps, 5 Seeds)

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

### Table 1B: Tier 2 Partner Record Study — Validation Verdicts, Training Speed (`tokensPerSec`), and End-of-Training $\mathbf{R}_B$ Diagnostics (1,000 Steps, 5 Seeds `3, 7, 11, 13, 17`)

| Arm / Configuration | Params | Val Mean | Val Tail Mean | Mean Tail Range | Paired $\Delta\text{Val}$ Mean $\pm s_\Delta$ | 95% Student-$t$ CI ($df=4$) | Tier 1 Verdict (Val) | Mean `tokensPerSec` | End $\|\mathbf{R}_A\|_2$ | End $\|\mathbf{R}_B\|_2$ | End $\cos(\mathbf{R}_A, \mathbf{R}_B)$ |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| **B0**: `scoutScale=0;rbMode=causal` *(default)* | 54,400 | **2.1742** | 2.1781 | 0.0099 | — *(baseline B0)* | — | **Baseline** | 1,305 | 5.1278 | 5.1572 | -0.7744 |
| **A1**: `recordB=off` (`scoutScale=0`) | 51,328 | 2.1769 | 2.1800 | 0.0077 | +0.0027 $\pm$ 0.0267 | `[-0.0305, +0.0359]` | **Inconclusive** | 1,398 | 4.8077 | 0.0000 | +0.0000 |
| **A2**: `rbMode=prevrev` (`scoutScale=0`) | 54,400 | 2.1742 | 2.1781 | 0.0099 | +0.0000 $\pm$ 0.0000 | `[+0.0000, +0.0000]` | **Inconclusive** | 706 | 5.1278 | 5.1572 | -0.7744 |
| **A3a**: `rbMode=lookback;lookback=16` (`scoutScale=0`) | 54,400 | 2.1766 | 2.1791 | 0.0098 | +0.0024 $\pm$ 0.0133 | `[-0.0141, +0.0188]` | **Inconclusive** | 927 | 5.2496 | 5.0972 | -0.7730 |
| **A3b**: `rbMode=lookback;lookback=64` (`scoutScale=0`) | 54,400 | 2.1796 | 2.1826 | 0.0111 | +0.0054 $\pm$ 0.0077 | `[-0.0042, +0.0150]` | **Inconclusive** | 657 | 5.2891 | 5.1642 | -0.8110 |
| **A4**: `scoutScale=0.5;scoutHead=separate` | 70,784 | 2.1945 | 2.1998 | 0.0115 | +0.0203 $\pm$ 0.0071 | `[+0.0114, +0.0292]` | **Degrades** | 540 | 5.2445 | 5.1755 | -0.8149 |
| v0.0014 default (`scoutScale=0.5;scoutHead=shared`) † | 54,400 | 2.2960 | 2.3021 | 0.0198 | +0.1219 $\pm$ 0.0151 | `[+0.1031, +0.1407]` | **Degrades** | 524 | 5.2328 *(s3)* | 5.2437 *(s3)* | -0.7437 *(s3)* |

### Table 1C: Partner Record $\mathbf{R}_B$ Norm $\|\mathbf{R}_B\|_2$ and Cosine $\cos(\mathbf{R}_A, \mathbf{R}_B)$ Trajectory Over Training Time (5-Seed Means across Steps `1 .. 1000` and Final Validation)

| Arm / Configuration | Step 1 $\|\mathbf{R}_B\| / \cos$ | Step 250 $\|\mathbf{R}_B\| / \cos$ | Step 500 $\|\mathbf{R}_B\| / \cos$ | Step 750 $\|\mathbf{R}_B\| / \cos$ | Step 1000 $\|\mathbf{R}_B\| / \cos$ | Final Val $\|\mathbf{R}_B\| / \cos$ |
|---|---|---|---|---|---|---|
| **B0**: `scoutScale=0;rbMode=causal` | 5.56 / -0.130 | 5.42 / -0.650 | 5.23 / -0.787 | 5.17 / -0.784 | 5.16 / -0.766 | **5.16 / -0.774** |
| **A1**: `recordB=off` | 0.00 / +0.000 | 0.00 / +0.000 | 0.00 / +0.000 | 0.00 / +0.000 | 0.00 / +0.000 | **0.00 / +0.000** |
| **A2**: `rbMode=prevrev` | 5.56 / -0.130 | 5.42 / -0.650 | 5.23 / -0.787 | 5.17 / -0.784 | 5.16 / -0.766 | **5.16 / -0.774** |
| **A3a**: `rbMode=lookback;lookback=16` | 5.66 / +0.071 | 5.42 / -0.635 | 5.21 / -0.766 | 5.11 / -0.775 | 5.10 / -0.766 | **5.10 / -0.773** |
| **A3b**: `rbMode=lookback;lookback=64` | 5.66 / +0.084 | 5.41 / -0.586 | 5.24 / -0.786 | 5.17 / -0.814 | 5.16 / -0.806 | **5.16 / -0.811** |
| **A4**: `scoutScale=0.5;scoutHead=separate` | 5.56 / -0.130 | 5.43 / -0.713 | 5.25 / -0.813 | 5.18 / -0.827 | 5.18 / -0.809 | **5.18 / -0.815** |
| v0.0014 default (`scoutScale=0.5;scoutHead=shared`, seed 3 repro) | 5.56 / -0.251 | 5.43 / -0.704 | 5.32 / -0.761 | 5.25 / -0.747 | 5.24 / -0.741 | **5.24 / -0.744** |

### Table 1D: Validation Learning-Rate Verification for Partner-Record Modes (`seed=3`, 1,000 Steps, `--stream=stride`, `--tail 5`, Validation Only)

| Partner-Record Mode | Candidate `lr` | Config String | Val Final | Val Tail Min | Val Tail Max | Val Tail Mean | Val Tail Range | Selected on Val? |
|---|---:|---|---:|---:|---:|---:|---:|---|
| **B0**: `scoutScale=0;rbMode=causal` † | `0.008` | `L12:arch=learns;dim=32;lr=0.008;scoutScale=0` † | 2.2434 | 2.2401 | 2.2463 | 2.2429 | 0.0062 | No |
| **B0**: `scoutScale=0;rbMode=causal` | `0.016` | `L12:arch=learns;dim=32;lr=0.016` | **2.1764** | 2.1732 | 2.1800 | 2.1765 | 0.0068 | **Yes (Winner)** |
| **B0**: `scoutScale=0;rbMode=causal` † | `0.024` | `L12:arch=learns;dim=32;lr=0.024;scoutScale=0` † | 2.2343 | 2.2324 | 2.2374 | 2.2347 | 0.0050 | No |
| **A3a**: `rbMode=lookback;lookback=16` | `0.008` | `L12:arch=learns;dim=32;lr=0.008;rbMode=lookback;lookback=16` | 2.2293 | 2.2282 | 2.2378 | 2.2327 | 0.0097 | No |
| **A3a**: `rbMode=lookback;lookback=16` | `0.016` | `L12:arch=learns;dim=32;lr=0.016;rbMode=lookback;lookback=16` | **2.1841** | 2.1785 | 2.1848 | 2.1829 | 0.0063 | **Yes (Winner)** |
| **A3a**: `rbMode=lookback;lookback=16` | `0.024` | `L12:arch=learns;dim=32;lr=0.024;rbMode=lookback;lookback=16` | 2.2587 | 2.2587 | 2.2774 | 2.2672 | 0.0187 | No |
| **A3b**: `rbMode=lookback;lookback=64` | `0.008` | `L12:arch=learns;dim=32;lr=0.008;rbMode=lookback;lookback=64` | 2.2424 | 2.2345 | 2.2449 | 2.2397 | 0.0104 | No |
| **A3b**: `rbMode=lookback;lookback=64` | `0.016` | `L12:arch=learns;dim=32;lr=0.016;rbMode=lookback;lookback=64` | **2.1865** | 2.1804 | 2.1917 | 2.1847 | 0.0113 | **Yes (Winner)** |
| **A3b**: `rbMode=lookback;lookback=64` | `0.024` | `L12:arch=learns;dim=32;lr=0.024;rbMode=lookback;lookback=64` | 2.2290 | 2.2289 | 2.2384 | 2.2318 | 0.0096 | No |
| **A4**: `scoutScale=0.5;scoutHead=separate` | `0.008` | `L12:arch=learns;dim=32;lr=0.008;scoutScale=0.5;scoutHead=separate` | 2.2489 | 2.2489 | 2.2588 | 2.2554 | 0.0099 | No |
| **A4**: `scoutScale=0.5;scoutHead=separate` | `0.016` | `L12:arch=learns;dim=32;lr=0.016;scoutScale=0.5;scoutHead=separate` | **2.2031** | 2.2028 | 2.2064 | 2.2038 | 0.0036 | **Yes (Winner)** |
| **A4**: `scoutScale=0.5;scoutHead=separate` | `0.024` | `L12:arch=learns;dim=32;lr=0.024;scoutScale=0.5;scoutHead=separate` | 2.2110 | 2.2093 | 2.2288 | 2.2159 | 0.0195 | No |

### Answers to the Three Core Tier 2 Questions (`Measured:` vs. `Hypothesis:`)

#### Question 1: Does $\mathbf{R}_B$ earn its parameters when the reverse loss is off? (**A1** vs. **B0**)
- **Measured:** **No — at 1,000 steps across 5 seeds, $\mathbf{R}_B$ does not provide a statistically detectable improvement over disabling $\mathbf{R}_B$ entirely (`Inconclusive`).** Disabling $\mathbf{R}_B$ (**A1**: `recordB=off`, which removes $\mathbf{W}_{\text{rec},B}$, $\mathbf{W}_{\text{cross},A}$, and $\mathbf{W}_{\text{cross},B}$, reducing parameters by 3,072 from `54,400` to `51,328` and increasing training speed by 7.1% from `1,305` to `1,398` tok/s) yields a 5-seed mean test loss of `2.2194` vs. `2.2181` for **B0** (`recordB=on, scoutScale=0`), a paired difference of $\Delta\text{Test} = +0.0013 \pm 0.0256$ nats/byte with 95% CI `[-0.0305, +0.0330]` (**Inconclusive**). On individual seeds, **A1** (`recordB=off`) beats **B0** on 3 of 5 seeds (seeds `3, 11, 17`: `2.1989 vs 2.2160`, `2.2124 vs 2.2251`, `2.2111 vs 2.2234`) and trails on 2 seeds (seeds `7, 13`: `2.2293 vs 2.2255`, `2.2454 vs 2.2006`).
- **Measured ($\mathbf{R}_B$ Norm & Cosine Trajectory):** As shown in Tables 1B and 1C, $\mathbf{R}_B$ in **B0** does not collapse to zero ($\|\mathbf{R}_B\|_2 = 5.1572$ vs. $\|\mathbf{R}_A\|_2 = 5.1278$ at validation) and does not become a positive copy of $\mathbf{R}_A$. Instead, $\cos(\mathbf{R}_A, \mathbf{R}_B)$ starts near `-0.130` at step 1 and becomes strongly negative (`-0.650` at step 250, `-0.787` at step 500, and `-0.7744` on final validation).
- **Hypothesis:** Because the frontier and core read $\mathbf{R}_{\text{combined}} = \frac{1}{2}(\mathbf{R}_A + \mathbf{R}_B)$ symmetrically, an anti-correlated partner record ($\cos(\mathbf{R}_A, \mathbf{R}_B) \approx -0.77$) acts primarily as a learned baseline subtraction / variance-reduction term against $\mathbf{R}_A$ rather than storing independent long-range context, which explains why a single record $\mathbf{R}_A$ (**A1**) achieves nearly the same loss (`2.2194` vs. `2.2181`) with 3,072 fewer parameters.

#### Question 2: Does a backward read of the recent past beat a forward carry? (**A2** and **A3** vs. **B0**)
- **Measured:** **No — neither chunk-boundary backward seeding (`rbMode=prevrev`, A2) nor rolling per-token backward recurrence over the last $N$ seen tokens (`rbMode=lookback`, A3a/A3b) improves on the forward causal carry B0 at 1,000 steps (`Inconclusive`), and both reduce training speed by 29%–50%.**
  - **A2 (`rbMode=prevrev`):** Yields 5-seed mean test loss `2.2181` vs. `2.2181` for **B0** ($\Delta\text{Test} = +0.0000 \pm 0.0000$ nats, 95% CI `[-0.0000, +0.0000]`, **Inconclusive**), while cutting training speed by 45.9% (`706` vs. `1,305` tok/s). Under `--stream=stride`, consecutive training chunks are non-contiguous, so `prevScoutRB` is reset at non-adjacent chunk boundaries during training (making training updates identical to **B0** while paying the cost of `scoutBackwardPass` on every step), and seeding $\mathbf{R}_B$ from `prevScoutRB` at the 878 contiguous chunk boundaries of `evalFullSplit` changes validation/test loss by at most $1.8\times 10^{-5}$ nats.
  - **A3a (`rbMode=lookback;lookback=16`):** Yields 5-seed mean test loss `2.2188` vs. `2.2181` for **B0** ($\Delta\text{Test} = +0.0007 \pm 0.0177$ nats, 95% CI `[-0.0213, +0.0227]`, **Inconclusive**), reducing training speed by 29.0% (`927` vs. `1,305` tok/s).
  - **A3b (`rbMode=lookback;lookback=64`):** Yields 5-seed mean test loss `2.2228` vs. `2.2181` for **B0** ($\Delta\text{Test} = +0.0046 \pm 0.0111$ nats, 95% CI `[-0.0091, +0.0184]`, **Inconclusive**), reducing training speed by 49.7% (`657` vs. `1,305` tok/s).
  - **Measured ($\mathbf{R}_B$ Behavior in A3):** Even though `rbMode=lookback` starts from $\mathbf{0}$ at every token $t$ and walks backwards over $x_t, x_{t-1}, \dots, x_{t-N+1}$ (so at step 1 $\cos(\mathbf{R}_A, \mathbf{R}_B)$ is slightly positive: `+0.071` for $N=16$ and `+0.084` for $N=64$), by step 500 backpropagating the forward loss through the backward recurrence drives $\cos(\mathbf{R}_A, \mathbf{R}_B)$ to `-0.766` ($N=16$) and `-0.786` ($N=64$), ending at `-0.7730` and `-0.8110` on final validation — nearly the same anti-correlated regime as causal carry **B0** (`-0.7744`).
- **Hypothesis:** In `rbMode=lookback`, because the backward walk starts at $x_t$ ($k=0$) and steps back to $x_{t-M+1}$ ($k=M-1$) with LayerNorm applied at every step $k$, the final state $\mathbf{R}_{B,t} = \mathbf{r}_B^{(M)}$ is most heavily weighted toward the *oldest* token in the window ($x_{t-M+1}$) while attenuating the most recent tokens ($x_t, x_{t-1}$) through $M$ successive LayerNorm contractions; consequently, longer lookback ($N=64$) dilutes local context more than $N=16$ (`2.2228` vs. `2.2188`) without adding predictive signal beyond what Component A's suffix recognizer and $\mathbf{R}_A$ already capture.

#### Question 3: Did `scoutScale=0.5` hurt because of the shared readout, or because reverse-LM gradients hurt the core itself? (**A4** vs. **B0** vs. **v0.0014 default**)
- **Measured:** **Both mechanisms hurt, and this experiment directly separates their magnitudes: 86.2% of the `scoutScale=0.5` test-loss penalty came from sharing the forward readout (`W_frontier` / `W_present`), while the remaining 13.8% (`+0.0170` nats) came from reverse-LM gradients flowing into the shared core and records.**
  1. **Total penalty of shared-head `scoutScale=0.5` vs. B0 (`scoutScale=0`):** Across 5 seeds at 1,000 steps, the v0.0014 default (`scoutScale=0.5;scoutHead=shared`) degrades test loss from `2.2181` (**B0**) to `2.3408` ($\Delta\text{Test} = +0.1227 \pm 0.0172$ nats, 95% CI `[+0.1013, +0.1441]`, **Degrades**).
  2. **Effect of isolating the scout readout head (A4 vs. v0.0014 default):** Giving the reverse-LM scout pass its own dedicated readout matrices (`W_frontier_scout`, `W_present_scout` in **A4**: `scoutScale=0.5;scoutHead=separate`) improves 5-seed test loss from `2.3408` to `2.2351` ($\Delta\text{Test} = -0.1057 \pm 0.0149$ nats, 95% CI `[-0.1241, -0.0873]`, $|\bar{\Delta}| = 0.1057 > \max\text{TailRange} = 0.0198$, **Improves**). Isolating the readout eliminates $0.1057 / 0.1227 = 86.2\%$ of the reverse-scout damage.
  3. **Residual penalty of reverse-LM gradients on the core and records (A4 vs. B0):** Even with a separate scout readout head, **A4** (`2.2351`) is still worse than **B0** (`scoutScale=0`, `2.2181`) on all 5 out of 5 seeds by $\Delta\text{Test} = +0.0170 \pm 0.0066$ nats (95% CI `[+0.0088, +0.0251]`, which excludes 0 and exceeds $\max\text{TailRange} = 0.0115$, **Degrades**), while cutting training speed by 58.6% (`540` vs. `1,305` tok/s). Thus, training the shared core and $\mathbf{R}_B$ to predict backward bytes degrades forward prediction by a smaller but statistically significant $+0.0170$ nats.

---

## 3. Tier 3: Full-Book Pass (`7,030` Steps) & Architecture Leaderboard

- **Measured (Tier 3 Decision Rule):** Because no Tier 2 partner-record variant achieved an **Improves** verdict over **B0** (`scoutScale=0;rbMode=causal`) at 1,000 steps under the Tier 1 rule (**A1**, **A2**, **A3a**, **A3b** are **Inconclusive**; **A4** **Degrades**), **B0** remains the best LEARNS configuration (and Tier 4 width scaling of `lookback` is skipped per the pre-registered rule).
- **Measured (Fresh v0.0015 Full-Pass Comparison):** Per Tier 3 instructions, we ran both **B0** (`L12:arch=learns;dim=32;lr=0.016`) and **`GRU ternary`** (`L12:arch=gru;dim=32;lr=0.016`) fresh under v0.0015 code (`codeHash: 0x626fd26c`) for one full pass (`7,030` steps = 1.0 epoch, `--stream=stride`, seeds `3, 7, 11`, `--tail 5 --final --curve 1000`).

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

### What Table 2A & Table 2B Show (`Measured:` vs. `Hypothesis:`)
- **Measured (LEARNS B0 vs. `GRU ternary` at 1,000 Steps vs. 7,030 Steps):**
  - At **1,000 steps across 5 seeds** (Table 1A), `GRU ternary` (`2.1096`) beats LEARNS **B0** (`2.2181`) by $\Delta\text{Test} = -0.1086 \pm 0.0053$ nats (95% CI `[-0.1152, -0.1020]`, **Improves**), winning on all 5 seeds by ~0.11 nats.
  - At **7,030 steps across 3 seeds** (Table 2A, both run fresh under v0.0015 code `0x626fd26c` with `--tail 5`), `GRU ternary` achieves a 3-seed mean test loss of `2.1218` (`2.1473 / 2.1263 / 2.0917`) vs. `2.1534` (`2.1657 / 2.1452 / 2.1493`) for LEARNS **B0** — a paired mean difference of $\Delta\text{Test} = -0.0316 \pm 0.0225$ nats in favor of `GRU ternary`, with 95% Student-$t$ CI `[-0.0875, +0.0243]` ($df=2$) and `GRU ternary` `meanTailRange = 0.0565`. Under the Tier 1 rule, this 3-seed full-pass comparison is **Inconclusive** (meaning 3 seeds with a 0.0565-nat ternary GRU tail jitter band cannot statistically resolve a ~0.032-nat difference; as noted in Rule 2, **Inconclusive** does not mean equivalent, and `GRU ternary` has the lower sample mean on all 3 seeds at 7,030 steps and is $2.9\times$ faster to train: `3,613` vs. `1,205` tok/s).
- **Measured (Quantization Gap vs. Architecture Gap at 7,030 Steps):**
  - In `float32` at 7,030 steps (`†`), `GRU float32 (lr=0.008)` reaches `1.7044` test nats/byte (`1.8638` at `lr=0.016`), whereas `LEARNS weights=float32;scoutScale=0 (lr=0.016)` reaches `1.9941` (`-0.1593` nats better than ternary LEARNS **B0** `2.1534`, and `0.0107` nats ahead of `Transformer float32` `2.0048`).
- **Hypothesis:** The remaining gap between LEARNS **B0** and the GRU (`-0.1086` nats in ternary at 1,000 steps; `-0.0316` nats in ternary and `-0.2897` nats in float32 at 7,030 steps) is driven by the GRU's per-dimension reset and update gates ($\mathbf{r}_t, \mathbf{z}_t \in (0,1)^H$ in every layer) versus LEARNS's un-gated additive residual accumulation $\mathbf{R}_{A,t} = \text{LN}(\mathbf{R}_{A,t-1} + Q(\mathbf{W}_{\text{rec},A})\mathbf{K}_{A,t} + Q(\mathbf{W}_{\text{cross},A})\mathbf{R}_{B,t-1})$ rather than by how $\mathbf{R}_B$ orders past tokens.
