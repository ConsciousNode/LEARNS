# LEARNS v0.0013 — Tier 2 Switchboard Reference (`SWITCHES.md`)

Every paper-derived component in LEARNS is controlled by an orthogonal switch exposed in three places:
1. **Browser Switchboard UI** (`SWITCHBOARD` tab in `index.html`, plus the `Settings String` box in the left panel).
2. **Compact Settings String** (`L12:arch=learns;dim=32;...`, accepted by `parseConfig()` and emitted by `serializeConfig()`).
3. **Headless CLI** (`node tests/run.js --config "key=val;..." --stream=stride|head --ngramData=all|seen`, `node tests/gradcheck.js key=val`, `node tests/optimizer-coverage.js key=val`).

With all switches at their default values and `--stream=head`, LEARNS v0.0013 is **bit-for-bit identical to v0.0011 and v0.0012** on the same seed, config, and corpus.

---

## Complete Switch Table

| Switch | Allowed Values | Default | Component / Paper | What `off` / Alternative Values Do | Active Parameter Count Change (`dim=d, layers=L, vocab=V`) |
|---|---|---|---|---|---|
| `arch` | `learns \| gru \| transformer \| unigram \| bigram \| trigram \| 5gram` | `learns` | Architecture selector | Selects LEARNS or one of the Tier 3 baselines in the same training/evaluation harness. | See `BASELINES.md` |
| `relEmbed` | `on \| off` | `on` | **(1) Relational Embedding** (*It's All Relative*) | `on`: $\mathbf{e}_t = \mathbf{W}_{\text{embed}}[x_t] + Q(\mathbf{W}_{\text{rel}})\mathbf{R}_{\text{comb},t-1}$. `off`: $\mathbf{e}_t = \mathbf{W}_{\text{embed}}[x_t]$; skips $\mathbf{W}_{\text{rel}}$ forward and backward. | $-d^2$ when `off` (`-1,024` at $d=32$) |
| `coreA` | `on \| off` | `on` | **(2) Temporal Turing Core A** (*The Electromagnetic Node*) | `on`: runs inner fixed point `innerFPSolve` over suffix signal $\mathbf{s}_t$. `off`: sets $\mathbf{a}_t = \mathbf{0}$ and cross-gate $\mathbf{g}_B = \mathbf{1}$. | $-L(2d^2 + 2d^2)$ when `off` (skips $\mathbf{W}_A, \mathbf{W}_{gA}, \mathbf{W}_{gB}$) |
| `coreB` | `on \| off` | `on` | **(2) Temporal Turing Core B** (*The Electromagnetic Node*) | `on`: $\mathbf{b}_t = \tanh(Q(\mathbf{W}_B)[\mathbf{h}_{t-1}; \mathbf{R}_{\text{comb},t-1}])$. `off`: sets $\mathbf{b}_t = \mathbf{0}$ and cross-gate $\mathbf{g}_A = \mathbf{1}$. | $-L(2d^2 + 2d^2)$ when `off` (skips $\mathbf{W}_B, \mathbf{W}_{gA}, \mathbf{W}_{gB}$) |
| `innerFP` | `2 .. 8` | `4` | **(2) Component A Inner FP** (*The Electromagnetic Node*) | Maximum iterations of $\mathbf{K}^{(i+1)} = \text{LN}(\tanh(Q(\mathbf{W}_A)[\mathbf{K}^{(i)}; \mathbf{s}_t]))$ from $\mathbf{K}^{(0)}=\mathbf{0}$. | `0` (shared $\mathbf{W}_A$ across iterations) |
| `fpIters` | `0 \| 1 \| 2 \| 6 \| 12` | `6` | **(3) Dyadic Fixed Point** (*K-H Fixed Point Theorem*) | Number of Jacobi iterations for $(\mathbf{K}_A, \mathbf{K}_B)$. `0` bypasses the Jacobi loop ($\mathbf{K}_A = \mathbf{g}_A \odot \mathbf{a}_t, \mathbf{K}_B = \mathbf{g}_B \odot \mathbf{b}_t$), running ~18% faster per step. | $-2Ld^2$ when `fpIters=0` (skips $\mathbf{W}_{FA}, \mathbf{W}_{FB}$) |
| `fpInject` | `off \| on` | `off` | **(3) Dyadic Fixed Point** (*K-H Fixed Point Theorem*) | `off`: autonomous map $\mathbf{K}_A' = \text{Norm}(\tanh(\mathbf{W}_{FA}\mathbf{K}_B))$. `on`: DEQ-style input injection $\mathbf{K}_A' = \text{Norm}(\tanh(\mathbf{W}_{FA}\mathbf{K}_B + \mathbf{a}_t))$. | `0` |
| `fpNorm` | `layernorm \| none \| rescale` | `layernorm` | **(3) Dyadic Fixed Point Contractivity** (*K-H Fixed Point Theorem*, v0.0013) | Controls normalization inside `dyadicFPSolve`:<br>• `layernorm`: standard LayerNorm (re-inflates vectors to radius $\sqrt{d}$ every step, preventing convergence).<br>• `none`: removes LayerNorm inside the Jacobi loop ($\mathbf{K}_A' = \tanh(\mathbf{W}_{FA}\mathbf{K}_B [+\mathbf{a}_t])$) and enforces spectral norm $\sigma(\mathbf{W}_{FA}), \sigma(\mathbf{W}_{FB}) \le 0.9$ via deterministic power iteration after each `optimizerStep`.<br>• `rescale`: centers $\tilde{\mathbf{x}} = \mathbf{x} - \mu$ and scales by $s = 1/\sqrt{1 + \|\tilde{\mathbf{x}}\|_2^2}$ without the $\sqrt{d}$ re-inflation ($\mathbf{y} = \boldsymbol{\gamma} \odot (s\tilde{\mathbf{x}}) + \boldsymbol{\beta}/\sqrt{d}$), enforcing $\|\boldsymbol{\gamma}\|_\infty \le 1$ and $\sigma(\mathbf{W}_{FA}), \sigma(\mathbf{W}_{FB}) \le 0.9$ so the centered Jacobi map is strictly contractive ($\text{Lip} \le 0.9 < 1$). | `0` (`none` leaves `ln_fpA/B` unused with 0 gradient) |
| `fpEps` | `1e-3 \| 1e-4 \| 1e-5` | `1e-4` | **(2 & 3) Fixed-Point Tolerance** | Early-stopping L2 residual threshold $\|\mathbf{K}^{(i+1)} - \mathbf{K}^{(i)}\|_2 < \varepsilon$. | `0` |
| `recordA` | `on \| off` | `on` | **(4) Record / Frontier Split** (*It's Time To Toss The Dice*) | `off`: clamps $\mathbf{R}_A = \mathbf{0}$ for all $t$, zeroes $\mathbf{W}_{\text{rec},A}$ and cross-coupling $\mathbf{W}_{\text{cross},B}$. | $-d^2$ (plus $-2d^2$ cross terms) |
| `recordB` | `on \| off` | `on` | **(4) Record / Frontier Split** (*It's Time To Toss The Dice*) | `off`: clamps $\mathbf{R}_B = \mathbf{0}$ for all $t$, skips the reverse-LM backward scout pass, and zeroes $\mathbf{W}_{\text{rec},B}$ and cross terms. | $-d^2$ (plus $-2d^2$ cross terms) |
| `rbMode` | `causal \| prevrev \| legacy` | `causal` | **(4) Partner Record Seeding** (*It's Time To Toss The Dice*) | `causal`: carries $\mathbf{R}_B$ forward across tokens. `prevrev`: seeds $\mathbf{R}_B$ at chunk start from `scoutBackwardPass` on the previous contiguous chunk (causal, leak-free). `legacy`: injects future scout $\mathbf{R}_B$ (leaks target; diagnostic only). | `0` |
| `frontierPresent` | `on \| off` | `on` | **(4) Frontier Readout** (*It's Time To Toss The Dice*) | `on`: $\text{logits}_t = Q(\mathbf{W}_{\text{frontier}})\mathbf{R}_{\text{comb},t} + Q(\mathbf{W}_{\text{present}})\mathbf{h}_{\text{frontier},t}$. `off`: record-only readout $Q(\mathbf{W}_{\text{frontier}})\mathbf{R}_{\text{comb},t}$. | $-Vd$ when `off` (`-8,192` at $d=32, V=256$) |
| `frontierHelix` | `on \| off` | `on` | **(4 & 5) Frontier Helix Path** | `on`: $\mathbf{h}_{\text{frontier},t} = \mathbf{h}_t^{(L)}$ (full damped rotating spiral). `off`: $\mathbf{h}_{\text{frontier},t} = \mathbf{P}_t^{(L)}$ (memoryless top-layer progression vector). | `0` |
| `helix` | `on \| off` | `on` | **(5) Helical State** (*The Drill of Entropy*) | `on`: $\mathbf{h}_t = \sigma(\boldsymbol{\rho}) \odot G(\boldsymbol{\theta})\mathbf{h}_{t-1} + \mathbf{P}_t$. `off`: zeroes recurrent carry ($\mathbf{h}_t = \mathbf{P}_t$). | $-L(d + d)$ when `off` (skips $\boldsymbol{\theta}, \boldsymbol{\rho}$) |
| `helixRotation` | `learned \| none` | `learned` | **(5) Givens Rotation** (*The Drill of Entropy*) | `learned`: rotates pair $(2i, 2i+1)$ by learned $\theta_i$. `none`: identity rotation $G(\mathbf{0}) = \mathbf{I}$ (decay only). | $-Ld$ when `none` |
| `helixRetention` | `learned \| fixed \| none` | `learned` | **(5) Helical Decay** (*The Drill of Entropy*) | `learned`: per-dim $\sigma(\rho_j)$ initialized at $\rho_j=2.2$. `fixed`: constant $\sigma(2.2) \approx 0.90025$. `none`: $\sigma = 1$ (undamped integration, v0.0009 style). | $-Ld$ when `fixed` or `none` |
| `thermo` | `hinge \| linear \| off` | `hinge` | **(6) Thermodynamic Loss** (*The Past Is Still The Future*) | `hinge`: $\mathcal{L}_{\text{thermo}} = \lambda \max(0, -\Delta S_{\text{chunk}})$. `linear`: $-\lambda \Delta S_{\text{chunk}}$. `off`: $0$. | `0` |
| `lambda` | $\ge 0$ (`float`) | `0.05` | **(6) Thermodynamic Weight** | Multiplier $\lambda$ on the chunk entropy penalty. | `0` |
| `weights` | `ternary \| float32` | `ternary` | **BitNet b1.58 Quantization** | `ternary`: $Q(\mathbf{W}) = \text{scale} \cdot \text{round\_clip}(\mathbf{W}/\text{scale}, -1, 1)$ with $\text{scale} = \text{mean}|\mathbf{W}|$ and straight-through estimator (STE). `float32`: unquantized dense matrices. | `0` (same parameter tensors; 1.58 bits vs 32 bits per weight) |
| `clipMode` | `global \| per_group` | `global` | **Gradient Norm Clipping** | `global`: clips joint L2 norm of all gradient tensors to $1.0$. `per_group`: clips each parameter tensor's L2 norm independently to $1.0$, isolating LayerNorm-bias gradient spikes. | `0` |
| `phiClip` | `on \| off` | `on` | **Gate Gradient Clamp** | `on`: clamps $\mathbf{W}_{gA,\text{grad}} \in [-\varphi, \varphi]$ and $\mathbf{W}_{gB,\text{grad}} \in [-1/\varphi, 1/\varphi]$ after norm clipping. Inert when `clipMode` caps L2 norm to $1.0$ ($1/\varphi \approx 0.618$). | `0` |

---

## Stream & Data-Matching Flags (`tests/run.js`, v0.0013)

| CLI Flag | Allowed Values | Default | Behaviour |
|---|---|---|---|
| `--stream` | `stride \| head` | `stride` | `stride`: deterministic per-epoch permutation of all $\lfloor(N_{\text{train}}-1)/\text{chunk}\rfloor = 7,030$ non-overlapping 64-byte chunk start positions (`buildStrideStarts`), seeded from `(seed, epoch)` via Mulberry32 without perturbing weight initialization. Resets recurrent state (`App.trainState = model.getEmptyState(); App.prevScoutRB = null;`) whenever consecutive chunks are not contiguous (`start !== prevStart + chunkSize`), identically across `LEARNS`, `GRU`, and `Transformer`.<br>`head`: exact v0.0012 sequential chunk start `start = (step * chunkSize) % (N_train - chunkSize - 2)`. |
| `--ngramData` | `all \| seen` | `all` | `all`: fits Kneser–Ney 1/2/3/5-gram tables on the entire 449,923-byte training split.<br>`seen`: builds the exact contiguous segments visited by the neural stream at the given `--steps`, `--stream`, and `--seed` (`buildSeenSegments`) and fits Kneser–Ney via `fitSegments()` on those exact target bytes without crossing non-adjacent chunk boundaries. |
| `--curve` | integer (e.g. `500`) | `0` (off) | Records causal validation loss (`valLoss`, `valBpb`) over the full validation split every `N` steps and at the final step in `record.valCurve`. |
| `--final` | boolean flag | `false` | Evaluates causal loss on the held-out 10% test split (`testLoss`, `testBpb`). Omitted during learning-rate sweeps so hyperparameters are chosen strictly on validation. |

---

## Verification

All 36 gradient configurations (`node tests/gradcheck.js --all`) and all 37 optimizer-coverage configurations (`node tests/optimizer-coverage.js --all`) pass with `ALL OK`.
