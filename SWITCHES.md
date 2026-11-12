# LEARNS v0.0016 — Component & Training Switches (`SWITCHES.md`)

Every architectural component and training mechanism in LEARNS v0.0016 (`index.html` and `tests/run.js`) is controlled by a key-value switch string serialized in canonical order (`L12:arch=...;dim=...`).

## Current Audited Default (**B0**, v0.0015 / v0.0016)

- `arch=learns;dim=32;layers=2;weights=ternary;relEmbed=on;coreA=on;coreB=on;innerFP=4;fpIters=6;fpEps=0.0001;fpInject=on;fpNorm=rescale;recordA=on;recordB=on;rbMode=causal;lookback=16;scoutScale=0;scoutHead=shared;frontierPresent=on;frontierHelix=on;helix=on;helixRotation=learned;helixRetention=learned;helixState=shared;thermo=hinge;lr=0.004;lrMin=0.00001;warmup=50;wd=0.01;clipMode=global;phiClip=on;ema=off`

## Switch Reference Table

| Switch | Values | Default | Paper / Component | Description |
| :--- | :--- | :---: | :--- | :--- |
| `arch` | `learns`, `gru`, `transformer`, `ngram` | `learns` | Harness | Model architecture |
| `dim` | integer (even) | `32` | All | Base embedding / state dimension $d$ |
| `layers` | integer | `2` | All | Number of stacked recurrent / attention layers |
| `weights` | `ternary`, `float32` | `ternary` | All | Absmean STE ternary quantization (`{-s, 0, +s}`) vs full float32 |
| `relEmbed` | `on`, `off` | `on` | *It's All Relative* (1) | Context-dependent relational shift $\mathbf{e}_t = \mathbf{W}_{\text{embed}}[x_t] + \mathbf{W}_{\text{rel}}\mathbf{R}_{\text{comb},t-1}$ |
| `coreA`, `coreB` | `on`, `off` | `on` | *The Electromagnetic Node* (2) | Enables/disables the $\mathbf{a}_t$ (suffix inner-FP) and $\mathbf{b}_t$ branches |
| `innerFP` | `0..12` | `4` | *The Electromagnetic Node* (2) | Inner fixed-point iterations for $\mathbf{a}_t$ |
| `fpIters` | `0..16` | `6` | *K-H Fixed Point Theorem* (3) | Dyadic cross-coupled fixed-point iterations per layer |
| `fpInject` | `on`, `off` | `on` | *K-H Fixed Point Theorem* (3) | Injects $(\mathbf{a}_t, \mathbf{b}_t)$ at every dyadic FP step |
| `fpNorm` | `rescale`, `layernorm`, `none` | `rescale` | *K-H Fixed Point Theorem* (3) | Norm inside dyadic FP solver (`rescale` eliminates LayerNorm division spikes) |
| `recordA`, `recordB` | `on`, `off` | `on` | *It's Time To Toss The Dice* (4) | Enables/disables forward record $\mathbf{R}_A$ and partner record $\mathbf{R}_B$ |
| `rbMode` | `causal`, `prevrev`, `lookback`, `window`, `legacy` | `causal` | *It's Time To Toss The Dice* (4) | Partner record update rule (`causal` carry; `prevrev` from prior chunk; `lookback` backward $N$-byte walk; `window` forward $N$-byte walk; `legacy` leaky scout) |
| `lookback` | `16`, `64` (integer $\ge 1$) | `16` | *It's Time To Toss The Dice* (4) | Window length $N$ for `rbMode=lookback` ($x_t \to x_{t-N+1}$) and `rbMode=window` ($x_{t-N+1} \to x_t$) |
| `scoutScale` | `0`, `0.5` (float $\ge 0$) | `0` | *It's Time To Toss The Dice* (4) | Weight on reverse-chunk auxiliary loss (`0` = disabled by default since v0.0015) |
| `scoutHead` | `shared`, `separate` | `shared` | *It's Time To Toss The Dice* (4) | Separate reverse readout matrices ($\mathbf{W}_{\text{frontier,scout}}, \mathbf{W}_{\text{present,scout}}$) when `scoutScale > 0` |
| `frontierPresent` | `on`, `off` | `on` | *It's Time To Toss The Dice* (4) | Includes present helical readout $\mathbf{W}_{\text{present}}\mathbf{h}_t$ in frontier logits |
| `frontierHelix` | `on`, `off` | `on` | *The Drill of Entropy* (5) | Reads full helical state $\mathbf{h}_t$ (`on`) vs unrotated proposal $\mathbf{P}_t$ (`off`) |
| `helix` | `on`, `off` | `on` | *The Drill of Entropy* (5) | Enables/disables Helical State update $\mathbf{h}_t = \sigma(\boldsymbol{\rho}) \odot \text{Rot}_{\boldsymbol{\theta}}(\mathbf{h}_{t-1}) + \mathbf{P}_t$ |
| `helixRotation` | `learned`, `none` | `learned` | *The Drill of Entropy* (5) | Learned pairwise Givens rotations $\boldsymbol{\theta}$ vs identity (`none`) |
| `helixRetention` | `learned`, `fixed`, `none` | `learned` | *The Drill of Entropy* (5) | Learned per-channel retention $\sigma(\boldsymbol{\rho})$ vs fixed $\sigma(2.2) \approx 0.90$ vs `1.0` (`none`) |
| `helixState` | `shared`, `per_layer` | `shared` | *The Drill of Entropy* (5) | Shared helical vector across layers vs per-layer $\mathbf{h}_t^{(\ell)}$ |
| `thermo` | `hinge`, `linear`, `off` | `hinge` | *The Past Is Still The Future* (6) | Thermodynamic record-entropy production penalty |
| `ema` | `on`, `off` | `off` | Signal & Noise (v0.0014) | Exponential moving average ($\beta=0.995$) of shadow weights for evaluation |
