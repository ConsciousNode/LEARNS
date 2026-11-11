# LEARNS v0.0015 ("The Partner Record") — Switchboard & Protocol Reference (`SWITCHES.md`)

Every paper-derived component and diagnostic control in LEARNS is controlled by an orthogonal switch exposed in three places:
1. **Browser Switchboard UI** (`SWITCHBOARD` tab in `index.html`, plus the `Settings String` box in the left panel).
2. **Compact Settings String** (`L12:arch=learns;dim=32;...`, accepted by `parseConfig()` and emitted by `serializeConfig()`).
3. **Headless CLI** (`node tests/run.js --config "key=val;..." --stream=stride|block|head --tail 5 --ngramData=all|seen`, `node tests/leak.js`, `node tests/gradcheck.js key=val`, `node tests/optimizer-coverage.js key=val`, `node tests/repro-v0011.js`, `node tests/tables.js`).

---

## Default Configuration vs. Legacy v0.0011 / v0.0012 / v0.0014 Bit-Identity

### 1. v0.0015 Default (`scoutScale=0;rbMode=causal;fpNorm=rescale;fpInject=on`)
In v0.0015, **`scoutScale=0` is promoted to the default** (selected by v0.0014's 5-seed Tier 4 result, where removing the reverse-chunk auxiliary loss improved 1,000-step test loss by $-0.1227 \pm 0.0172$ nats and 7,030-step test loss by $-0.1465 \pm 0.0156$ nats under the Tier 1 rule). Combined with `fpNorm=rescale;fpInject=on` (promoted in v0.0014), the v0.0015 default is **B0** (`scoutScale=0;rbMode=causal;fpNorm=rescale;fpInject=on`).

### 2. Explicit Legacy v0.0011 / v0.0012 and v0.0014 Configurations
To reproduce LEARNS v0.0011 and v0.0012 **bit-for-bit** (40 steps at 12-decimal precision and 150 steps under `--stream=head`), or to reproduce the v0.0014 default (`scoutScale=0.5`, 1,000 steps under `--stream=stride`) **bit-for-bit**, pass `scoutScale=0.5` explicitly or run:

```bash
node tests/repro-v0011.js
```

---

## Complete Switch Table

| Switch | Allowed Values | v0.0015 Default | Component / Paper | What `off` / Alternative Values Do | Active Parameter Count Change (`dim=d, layers=L, vocab=V`) |
|---|---|---|---|---|---|
| `arch` | `learns \| gru \| transformer \| unigram \| bigram \| trigram \| 5gram` | `learns` | Architecture selector | Selects LEARNS or one of the Tier 3 baselines in the same training/evaluation harness. | See `BASELINES.md` |
| `relEmbed` | `on \| off` | `on` | **(1) Relational Embedding** (*It's All Relative*) | `on`: $\mathbf{e}_t = \mathbf{W}_{\text{embed}}[x_t] + Q(\mathbf{W}_{\text{rel}})\mathbf{R}_{\text{comb},t-1}$. `off`: $\mathbf{e}_t = \mathbf{W}_{\text{embed}}[x_t]$; skips $\mathbf{W}_{\text{rel}}$ forward and backward. | $-d^2$ when `off` (`-1,024` at $d=32$) |
| `coreA` | `on \| off` | `on` | **(2) Temporal Turing Core A** (*The Electromagnetic Node*) | `on`: runs inner fixed point `innerFPSolve` over suffix signal $\mathbf{s}_t$. `off`: sets $\mathbf{a}_t = \mathbf{0}$ and cross-gate $\mathbf{g}_B = \mathbf{1}$. | $-L(2d^2 + 2d^2)$ when `off` (skips $\mathbf{W}_A, \mathbf{W}_{gA}, \mathbf{W}_{gB}$) |
| `coreB` | `on \| off` | `on` | **(2) Temporal Turing Core B** (*The Electromagnetic Node*) | `on`: $\mathbf{b}_t = \tanh(Q(\mathbf{W}_B)[\mathbf{h}_{\text{in},t}; \mathbf{R}_{\text{comb},t-1}])$. `off`: sets $\mathbf{b}_t = \mathbf{0}$ and cross-gate $\mathbf{g}_A = \mathbf{1}$. | $-L(2d^2 + 2d^2)$ when `off` (skips $\mathbf{W}_B, \mathbf{W}_{gA}, \mathbf{W}_{gB}$) |
| `innerFP` | `2 .. 8` | `4` | **(2) Component A Inner FP** (*The Electromagnetic Node*) | Maximum iterations of $\mathbf{K}^{(i+1)} = \text{LN}(\tanh(Q(\mathbf{W}_A)[\mathbf{K}^{(i)}; \mathbf{s}_t]))$ from $\mathbf{K}^{(0)}=\mathbf{0}$. | `0` (shared $\mathbf{W}_A$ across iterations) |
| `fpIters` | `0 \| 1 \| 2 \| 6 \| 12` | `6` | **(3) Dyadic Fixed Point** (*K-H Fixed Point Theorem*) | Number of Jacobi iterations for $(\mathbf{K}_A, \mathbf{K}_B)$. `0` bypasses the Jacobi loop ($\mathbf{K}_A = \mathbf{g}_A \odot \mathbf{a}_t, \mathbf{K}_B = \mathbf{g}_B \odot \mathbf{b}_t$). | $-2Ld^2$ when `fpIters=0` (skips $\mathbf{W}_{FA}, \mathbf{W}_{FB}$) |
| `fpInject` | `on \| off` | `on` | **(3) Dyadic Fixed Point** (*K-H Fixed Point Theorem*, v0.0014 default `on`) | `on`: DEQ-style input injection $\mathbf{K}_A' = \text{Norm}(\tanh(\mathbf{W}_{FA}\mathbf{K}_B + \mathbf{a}_t))$ inside the Jacobi loop. `off`: autonomous map $\mathbf{K}_A' = \text{Norm}(\tanh(\mathbf{W}_{FA}\mathbf{K}_B))$. | `0` |
| `fpNorm` | `rescale \| none \| layernorm` | `rescale` | **(3) Dyadic Fixed Point Contractivity** (*K-H Fixed Point Theorem*, v0.0014 default `rescale`) | Controls normalization inside `dyadicFPSolve`:<br>• `rescale` *(default)*: centers $\tilde{\mathbf{x}} = \mathbf{x} - \mu$ and scales by $s = 1/\sqrt{1 + \|\tilde{\mathbf{x}}\|_2^2}$ without the $\sqrt{d}$ re-inflation ($\mathbf{y} = \boldsymbol{\gamma} \odot (s\tilde{\mathbf{x}}) + \boldsymbol{\beta}/\sqrt{d}$), enforcing $\|\boldsymbol{\gamma}\|_\infty \le 1$ and $\sigma(\mathbf{W}_{FA}), \sigma(\mathbf{W}_{FB}) \le 0.9$ so the centered Jacobi map is strictly contractive ($\text{Lip} \le 0.9 < 1$).<br>• `none`: removes LayerNorm inside the Jacobi loop and enforces $\sigma(\mathbf{W}_{FA}), \sigma(\mathbf{W}_{FB}) \le 0.9$.<br>• `layernorm`: legacy LayerNorm inside the Jacobi loop. | `0` (`none` leaves `ln_fpA/B` unused with 0 gradient) |
| `fpEps` | `1e-3 \| 1e-4 \| 1e-5` | `1e-4` | **(2 & 3) Fixed-Point Tolerance** | Early-stopping L2 residual threshold $\|\mathbf{K}^{(i+1)} - \mathbf{K}^{(i)}\|_2 < \varepsilon$. | `0` |
| `recordA` | `on \| off` | `on` | **(4) Record / Frontier Split** (*It's Time To Toss The Dice*) | `off`: clamps $\mathbf{R}_A = \mathbf{0}$ for all $t$, zeroes $\mathbf{W}_{\text{rec},A}$ and cross-coupling $\mathbf{W}_{\text{cross},B}$. | $-d^2$ (plus $-2d^2$ cross terms) |
| `recordB` | `on \| off` | `on` | **(4) Record / Frontier Split** (*It's Time To Toss The Dice*, Arm A1) | `off`: clamps $\mathbf{R}_B = \mathbf{0}$ for all $t$, skips the reverse-LM backward scout pass, and zeroes $\mathbf{W}_{\text{rec},B}$ and cross terms $\mathbf{W}_{\text{cross},A}, \mathbf{W}_{\text{cross},B}$. | $-3d^2$ when `off` (`-3,072` at $d=32$: `54,400 -> 51,328`) |
| `rbMode` | `causal \| lookback \| prevrev \| legacy` | `causal` | **(4) Partner Record Mode** (*It's Time To Toss The Dice*, Arms B0, A2, A3) | • `causal` *(default, B0)*: carries $\mathbf{R}_B$ forward causally across tokens.<br>• `lookback` *(v0.0015 Arm A3)*: at each token $t$, computes $\mathbf{R}_{B,t}$ by running the $\mathbf{R}_B$ recurrence backwards over the last $N = \text{lookback}$ already-seen tokens $x_t, x_{t-1}, \dots, x_{\max(0, t-N+1)}$ starting from $\mathbf{0}$ (clamped at stream/chunk start), with exact BPTT through the backward recurrence.<br>• `prevrev` *(Arm A2)*: seeds $\mathbf{R}_B$ at chunk start from `scoutBackwardPass` on the previous contiguous chunk.<br>• `legacy`: injects future scout $\mathbf{R}_B$ (leaks target; positive control in `tests/leak.js`). | `0` |
| `lookback` | integer $N \ge 1$ (`16 \| 64`) | `16` | **(4) Rolling Look-Back Window $N$** (*It's Time To Toss The Dice*, v0.0015 Arm A3) | Number of already-seen past tokens $x_t, x_{t-1}, \dots, x_{t-N+1}$ walked backwards from $\mathbf{0}$ to form $\mathbf{R}_{B,t}$ when `rbMode=lookback`. Ignored when `rbMode != lookback`. | `0` |
| `scoutScale` | `0 \| 0.5` | `0` | **(4) Auxiliary Reverse-Scout Gradient Weight** (*It's Time To Toss The Dice*, **v0.0015 new default `0`**) | Multiplier on the reverse-chunk auxiliary loss gradients (`scoutBackwardPass`, predicting $x_{t-1}$ from $x_t$) in `trainChunk`.<br>• `0` *(v0.0015 default)*: keeps both $\mathbf{R}_A$ and $\mathbf{R}_B$ active causally (`54,400` params) without reverse-LM gradients.<br>• `0.5` *(v0.0011–v0.0014 legacy)*: adds $0.5 \nabla \mathcal{L}_{\text{rev}}$ into the readout and core. | `0` when `scoutHead=shared` |
| `scoutHead` | `shared \| separate` | `shared` | **(4) Reverse-Scout Readout Head** (*It's Time To Toss The Dice*, v0.0015 Arm A4) | Controls which readout matrices `scoutBackwardPass` uses when `scoutScale > 0`:<br>• `shared` *(default)*: uses the forward readout $\mathbf{W}_{\text{frontier}}, \mathbf{W}_{\text{present}}$.<br>• `separate` *(Arm A4)*: allocates dedicated $\mathbf{W}_{\text{frontier,scout}}, \mathbf{W}_{\text{present,scout}} \in \mathbb{R}^{V \times d}$ so reverse-LM gradients train $\mathbf{R}_B$ and the core without touching the forward readout. | $+2Vd$ during training when `separate` and `scoutScale > 0` (`+16,384` at $d=32, V=256$; `54,400` inference params) |
| `frontierPresent` | `on \| off` | `on` | **(4) Frontier Readout** (*It's Time To Toss The Dice*) | `on`: $\text{logits}_t = Q(\mathbf{W}_{\text{frontier}})\mathbf{R}_{\text{comb},t} + Q(\mathbf{W}_{\text{present}})\mathbf{h}_{\text{frontier},t}$. `off`: record-only readout $Q(\mathbf{W}_{\text{frontier}})\mathbf{R}_{\text{comb},t}$. | $-Vd$ when `off` (`-8,192` at $d=32, V=256$) |
| `frontierHelix` | `on \| off` | `on` | **(4 & 5) Frontier Helix Path** | `on`: $\mathbf{h}_{\text{frontier},t} = \mathbf{h}_t^{(L)}$ (full damped rotating spiral). `off`: $\mathbf{h}_{\text{frontier},t} = \mathbf{P}_t^{(L)}$ (memoryless top-layer progression vector). | `0` |
| `helix` | `on \| off` | `on` | **(5) Helical State** (*The Drill of Entropy*) | `on`: $\mathbf{h}_t = \sigma(\boldsymbol{\rho}) \odot G(\boldsymbol{\theta})\mathbf{h}_{t-1} + \mathbf{P}_t$. `off`: zeroes recurrent carry ($\mathbf{h}_t = \mathbf{P}_t$). | $-L(d + d)$ when `off` (skips $\boldsymbol{\theta}, \boldsymbol{\rho}$) |
| `helixRotation` | `learned \| none` | `learned` | **(5) Givens Rotation** (*The Drill of Entropy*) | `learned`: rotates pair $(2i, 2i+1)$ by learned $\theta_i$. `none`: identity rotation $G(\mathbf{0}) = \mathbf{I}$ (decay only). | $-Ld$ when `none` |
| `helixRetention` | `learned \| fixed \| none` | `learned` | **(5) Helical Decay** (*The Drill of Entropy*) | `learned`: per-dim $\sigma(\rho_j)$ initialized at $\rho_j=2.2$. `fixed`: constant $\sigma(2.2) \approx 0.90025$. `none`: $\sigma = 1$ (undamped integration). | $-Ld$ when `fixed` or `none` |
| `helixState` | `shared \| per_layer` | `shared` | **(5) Helical Recurrent State Capacity** (*The Drill of Entropy*, v0.0014 Tier 3.2) | `shared`: carries a single $d$-float helical vector $\mathbf{h}_t$ across tokens (`3d = 96` total state floats at $d=32$).<br>`per_layer`: carries a dedicated helical state vector $\mathbf{h}_t^{(l)} \in \mathbb{R}^d$ per layer (`(2+L)d = 128` total state floats at $d=32, L=2$). | `0` (`+32` recurrent state floats at $d=32, L=2$) |
| `thermo` | `hinge \| linear \| off` | `hinge` | **(6) Thermodynamic Loss** (*The Past Is Still The Future*) | `hinge`: $\mathcal{L}_{\text{thermo}} = \lambda \max(0, -\Delta S_{\text{chunk}})$. `linear`: $-\lambda \Delta S_{\text{chunk}}$. `off`: $0$. | `0` |
| `lambda` | $\ge 0$ (`float`) | `0.05` | **(6) Thermodynamic Weight** | Multiplier $\lambda$ on the chunk entropy penalty. | `0` |
| `weights` | `ternary \| float32` | `ternary` | **BitNet b1.58 Quantization** | `ternary`: $Q(\mathbf{W}) = \text{scale} \cdot \text{round\_clip}(\mathbf{W}/\text{scale}, -1, 1)$ with $\text{scale} = \text{mean}|\mathbf{W}|$ and straight-through estimator (STE). `float32`: unquantized dense matrices. | `0` (same parameter tensors; 1.58 bits vs 32 bits per weight) |
| `clipMode` | `global \| per_group` | `global` | **Gradient Norm Clipping** | `global`: clips joint L2 norm of all gradient tensors to $1.0$. `per_group`: clips each parameter tensor's L2 norm independently to $1.0$. | `0` |
| `phiClip` | `on \| off` | `on` | **Gate Gradient Clamp** | `on`: clamps $\mathbf{W}_{gA,\text{grad}} \in [-\varphi, \varphi]$ and $\mathbf{W}_{gB,\text{grad}} \in [-1/\varphi, 1/\varphi]$ after norm clipping. | `0` |
| `ema` | `off \| on` | `off` | **EMA Shadow Weight Evaluation** (v0.0014 Tier 1) | `on`: maintains an exponential moving average ($\beta_{\text{ema}} = 0.99$) of all float shadow parameter arrays after each `optimizerStep`, swapping and quantizing the EMA weights once at evaluation time (`evalFullSplit`). | `0` |

---

## Verification

- `node tests/leak.js`: `ALL OK` (`max |Δlogit| === 0` across all causal modes on random and trained weights; `rbMode=legacy` fails as positive control; `rbMode=lookback` scratch equivalence `=== 0`).
- `node tests/repro-v0011.js`: `ALL OK` (40-step v0.0011, 150-step v0.0012 `--stream=head`, and 1,000-step v0.0014 `scoutScale=0.5` reproduce bit-for-bit).
- `node tests/gradcheck.js --all`: `ALL OK (45 configurations verified)`.
- `node tests/optimizer-coverage.js --all`: `ALL OK (46 configurations verified)`.
