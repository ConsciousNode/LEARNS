# LEARNS v0.0013 — The Long Run

**Logical Entropic Arrow Recursive Neuro-Symbolic Architecture**  
*ConsciousNode SoftWorks — Single-File Browser Language Model & Honest Experimental Lab*

- **Code Hash (`CODE_HASH`)**: `0xdf8ae499`
- **Corpus Hash (`CORPUS_HASH`)**: `0x39c84ca9` (`corpus.txt`, *The Adventures of Sherlock Holmes*, 562,404 ASCII bytes; deterministic 80/10/10 contiguous byte split: `449,923` train / `56,240` validation / `56,241` test bytes)

---

## Full-Pass Leaderboard (`7,030` Steps = 1 Full Epoch Over Train Split, `dim=32`, `--stream=stride`, Seeds `3 / 7 / 11`)

Every neural contender below was trained for **one full pass (`7,030` steps × 64 bytes = `449,920` target bytes)** over the 80% training split using `--stream=stride` (a deterministic seeded permutation of all 7,030 non-overlapping 64-byte chunks with clean recurrent state reset on non-contiguous chunks), after re-sweeping learning rates on validation (`seed=3`). Kneser–Ney $n$-gram baselines are shown both fitted on the full split (`--ngramData=all`) and data-matched to the exact segments seen by the neural models (`--ngramData=seen`):

| Rank | Architecture / Model (`dim=32`, `7,030` steps, `--stream=stride`) | Active Params | Mean Val Loss (nats/B) | Mean Test Loss (nats/B) | Min–Max Test Loss | Seeds (`3 / 7 / 11`) | Mean Test bits/B |
|---|---|---:|---:|---:|---:|---|---:|
| 1 | **Kneser–Ney 5-gram (`all` / `seen`)** | 123,785 / 121,311 | 1.4064 / 1.4104 | **1.4432 / 1.4462** | 1.4462–1.4463 | 1.4462 / 1.4463 / 1.4462 | **2.0821 / 2.0864** |
| 2 | **GRU (`weights=float32`, `lr=0.008`)** | 54,450 | 1.6640 | **1.7044** | 1.6914–1.7207 | 1.7010 / 1.7207 / 1.6914 | **2.4589** |
| 3 | **Kneser–Ney 3-gram (`all` / `seen`)** | 11,042 / 10,985 | 1.9003 / 1.9007 | **1.9358 / 1.9359** | 1.9359–1.9360 | 1.9359 / 1.9360 / 1.9359 | **2.7928 / 2.7929** |
| 4 | **Transformer (`weights=float32`, `lr=0.008`)** | 55,440 | 1.9643 | **2.0048** | 1.9960–2.0155 | 2.0029 / 1.9960 / 2.0155 | **2.8923** |
| 5 | **GRU (`weights=ternary`, `lr=0.016`)** | 54,450 | 2.0746 | **2.1218** | 2.0917–2.1473 | 2.1473 / 2.1263 / 2.0917 | **3.0610** |
| 6 | **Transformer (`weights=ternary`, `lr=0.016`)** | 55,440 | 2.1034 | **2.1408** | 2.1123–2.1822 | 2.1279 / 2.1822 / 2.1123 | **3.0886** |
| 7 | **LEARNS `relEmbed=off` (`weights=ternary`, `lr=0.016`)** | 53,376 | 2.3083 | **2.3472** | 2.3223–2.3751 | 2.3751 / 2.3442 / 2.3223 | **3.3863** |
| 8 | **LEARNS default (`weights=ternary`, `lr=0.016`)** | 54,400 | 2.3366 | **2.3753** | 2.3497–2.3996 | 2.3497 / 2.3766 / 2.3996 | **3.4268** |
| 9 | **Kneser–Ney 2-gram (`all` / `seen`)** | 1,568 | 2.4034 | **2.4349** | 2.4349–2.4349 | 2.4349 / 2.4349 / 2.4349 | **3.5128** |
| 10 | **Kneser–Ney 1-gram (`all` / `seen`)** | 78 | 3.0785 | **3.0953** | 3.0953–3.0953 | 3.0953 / 3.0953 / 3.0953 | **4.4656** |

---

## Convergent Dyadic Fixed Point (`fpNorm=layernorm|none|rescale`)

In v0.0011 and v0.0012 (`fpNorm=layernorm`), the dyadic fixed point never converged because LayerNorm re-inflated $(\mathbf{K}_A, \mathbf{K}_B)$ to radius $\sqrt{d}$ on every iteration (`fpResidual = 1.28–7.07`) and produced `4.2e8` LayerNorm-bias gradient spikes. Replacing LayerNorm inside the Jacobi loop with contractive normalization (`fpNorm=none` or `fpNorm=rescale`, with $\sigma(\mathbf{W}_{FA}), \sigma(\mathbf{W}_{FB}) \le 0.9$ enforced via power iteration after each `optimizerStep`) lets the fixed point actually converge — and when paired with input injection (`fpInject=on`), yields the best LEARNS accuracy in the study:

| Configuration (`dim=32, lr=0.016, --stream=stride`) | 1,000-Step 3-Seed Mean FP Residual | 1,000-Step Mean FP Iters | 1,000-Step 3-Seed Mean Test Loss (`3 / 7 / 11`) | 3,515-Step (Half-Pass, `seed=3`) Test Loss |
|---|---:|---:|---|---:|
| **`fpNorm=layernorm` (`fpInject=off`, default)** | `7.067e+0` | `5.95` | `2.4509` (2.3653 / 2.5381 / 2.4494) | `2.3152` (`449.9s`) |
| **`fpNorm=rescale` (`fpInject=off`)** | **`1.636e-3`** | `6.00` | **`2.3733`** (2.4027 / 2.3912 / 2.3260) | **`2.3104`** (`398.6s`) |
| **`fpNorm=none` (`fpInject=off`)** | **`1.125e-1`** | `6.00` | **`2.3681`** (2.3359 / 2.3707 / 2.3977) | **`2.3068`** (`430.8s`) |
| **`fpNorm=none;fpInject=on`** | **`1.350e-3`** | `6.00` | **`2.3565`** (2.3536 / 2.3369 / 2.3791) | **`2.2758`** (**`348.7s`**) |
| **`fpNorm=rescale;fpInject=on`** | **`2.033e-5`** | **`4.32`** | **`2.3460`** (2.3657 / 2.3325 / 2.3398) | **`2.2929`** (`450.3s`) |

---

## Six Components & Empirical Verdict Summary

| # | Component | Paper Attribution | Switches | Empirical Verdict in v0.0013 (`--stream=stride`) |
|---|---|---|---|---|
| 1 | **Relational Embedding** | *It's All Relative* | `relEmbed=on\|off` | **Hurts early (`-0.071` nats at 1,000 steps), neutral/helpful at full pass**: penalty shrinks to `-0.028` nats across 3 seeds at 7,030 steps, and `relEmbed=on` beats `relEmbed=off` by **`+0.025` nats on `seed=3` at 7,030 steps** (`2.3497` vs `2.3751`) once records mature. |
| 2 | **Temporal Turing Core** | *The Electromagnetic Node* | `coreA=on\|off`, `coreB=on\|off`, `innerFP=2..8` | **Helps on `seed=3` (`+0.030` nats for `coreA`, `+0.023` nats for `coreB` at 1,000 steps)**; mixed across seeds under `fpNorm=layernorm` due to Jacobi LN spikes. |
| 3 | **Dyadic Fixed Point** | *K-H Fixed Point Theorem* | `fpIters=0\|1\|2\|6\|12`, `fpInject=off\|on`, `fpNorm=layernorm\|none\|rescale` | **Hurts when non-convergent (`fpNorm=layernorm`, where `fpIters=0` is `-0.106` nats better and ~18% faster); HELPS when contractive + input-injected (`fpNorm=rescale;fpInject=on` `-0.105` nats at 1,000 steps; `fpNorm=none;fpInject=on` reaches `2.2758` at 3,515 steps)**. |
| 4 | **Record / Frontier Split** | *It's Time To Toss The Dice* | `recordA=on\|off`, `recordB=on\|off`, `rbMode=causal\|prevrev\|legacy`, `frontierPresent=on\|off`, `frontierHelix=on\|off` | **`frontierPresent` is Load-Bearing #1 (`+0.655` nats when turned off)**. Without `W_present` reading the working state, the record-only frontier collapses to unigram loss (`3.1061`). |
| 5 | **Helical State** | *The Drill of Entropy* | `helix=on\|off`, `helixRotation=learned\|none`, `helixRetention=learned\|fixed\|none` | **`helix` is Load-Bearing #2 (`+0.422` nats when turned off)**. Learned decay $\sigma(\boldsymbol{\rho})$ prevents the undamped integration slowdown on `seed=3` (`+0.040` nats). |
| 6 | **Thermodynamic Loss** | *The Past Is Still The Future* | `thermo=hinge\|linear\|off`, `lambda` | **Helps on `seed=3` (`+0.033` nats when turned off at 1,000 steps)**; mixed across seeds under `fpNorm=layernorm`. |

---

## Reproducing Any Number from the Command Line

1. **Verify exact gradients and optimizer coverage (including `fpNorm=layernorm|none|rescale`)**:
   ```bash
   node tests/gradcheck.js --all
   node tests/optimizer-coverage.js --all
   ```
2. **Reproduce v0.0011 / v0.0012 sequential runs (`--stream=head`)**:
   ```bash
   node tests/run.js --arch learns --config "dim=32;lr=0.004" --seed 3 --steps 150 --stream head --final
   node tests/train-probe.js corpus.txt 40 3
   ```
3. **Run full-pass (`7,030`-step) `--stream=stride` training with 500-step validation curves**:
   ```bash
   node tests/run.js --arch learns --config "dim=32;lr=0.016" --seed 3 --steps 7030 --stream stride --curve 500 --final
   node tests/run.js --arch gru --config "dim=32;lr=0.008;weights=float32" --seed 3 --steps 7030 --stream stride --curve 500 --final
   ```
4. **Run contractive fixed-point (`fpNorm=none` / `fpNorm=rescale`) and data-matched Kneser–Ney (`--ngramData=seen`)**:
   ```bash
   node tests/run.js --arch learns --config "dim=32;lr=0.016;fpNorm=none;fpInject=on" --seed 3 --steps 3515 --stream stride --final
   node tests/run.js --arch learns --config "dim=32;lr=0.016;fpNorm=rescale;fpInject=on" --seed 3 --steps 1000 --stream stride --final
   node tests/run.js --arch 5gram --ngramData seen --stream stride --steps 150 --seed 3 --final
   ```

See `EXPERIMENT_REPORT.md` for full learning curves, diagnostic tables, and analysis, `SWITCHES.md` for the switchboard reference, and `BASELINES.md` for baseline equations and parameter-matching formulas.
