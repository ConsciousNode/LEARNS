# LEARNS v0.0012 — Switchboard Reference (`SWITCHES.md`)

Every paper-derived component and numerical policy in LEARNS v0.0012 is exposed as a switch in the **Switchboard** UI panel, the `.learns` save format, the `L12:...` settings string, `tests/run.js`, `tests/gradcheck.js`, and `tests/optimizer-coverage.js`.

When all switches are left at their defaults, LEARNS v0.0012 is **bit-for-bit identical** to LEARNS v0.0011 (`CODE_HASH: 0xfe2cf657`, `CORPUS_HASH: 0x39c84ca9`). Disabling any component removes its computational path cleanly (via an exact zero or identity bypass with matching analytical backward pass) rather than merely freezing its weights.

## Complete Switch Table

| Group | Switch Key | Default | Options | Theory Paper Tested | What It Controls & Clean Bypass Behavior | Active Param Delta (`dim=32, L=2, V=256`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Relational Embedding** | `relEmbed` | `on` | `on \| off` | *It's All Relative* | **`on`**: $e_t = \text{embed}(x_t) + Q(W_{\text{rel}})\,R_{\text{comb},t-1}$. **`off`**: $e_t = \text{embed}(x_t)$ alone; $W_{\text{rel}}$ is bypassed in forward and backward. | $-1,024$ (`53,376`) |
| **B. Temporal Turing Core** | `coreA` | `on` | `on \| off` | *The Electromagnetic Node* | **`on`**: Component A runs inner fixed-point solver over `suffix_sig` ($a_t = \text{InnerFP}(\text{suffix\_sig})$). **`off`**: $a_t = 0$, $K_{A,\text{init}} = 0$, cross-gates $W_{gA}, W_{gB}$ bypassed ($g_B = 1$). | $-10,240$ (`44,160`) |
| **B. Temporal Turing Core** | `coreB` | `on` | `on \| off` | *The Electromagnetic Node* | **`on`**: Component B computes $b_t = \tanh(Q(W_B)[h, R_{\text{comb},t-1}])$. **`off`**: $b_t = 0$, $K_{B,\text{init}} = 0$, cross-gates $W_{gA}, W_{gB}$ bypassed ($g_A = 1$). | $-8,192$ (`46,208`) |
| **B. Temporal Turing Core** | `innerFP` | `4` | `2..8` (`2 \| 3 \| 4 \| 6 \| 8`) | *The Electromagnetic Node* | Maximum iterations of Component A's inner fixed-point loop $K^{(i+1)} = \text{LN}(\tanh(Q(W_A)[K^{(i)}, \text{sfx}]))$. | $0$ (`54,400`) |
| **C. Dyadic Fixed Point** | `fpIters` | `6` | `0 \| 1 \| 2 \| 6 \| 12` | *K-H Fixed Point Theorem* | Number of Jacobi iterations between $K_A$ and $K_B$. **`0`**: no dyadic iteration ($K_A = K_{A,\text{init}}, K_B = K_{B,\text{init}}$; single feed-forward pass, $W_{FA}, W_{FB}, \text{LN}_{\text{fpA/B}}$ bypassed). | $-4,096$ at `0` (`50,304`) |
| **C. Dyadic Fixed Point** | `fpInject` | `off` | `off \| on` | *K-H Fixed Point Theorem* | **`off`**: autonomous Jacobi map $K_A' = \text{LN}(\tanh(W_{FA}K_B))$. **`on`**: DEQ-style input injection $K_A' = \text{LN}(\tanh(W_{FA}K_B + a_t))$, $K_B' = \text{LN}(\tanh(W_{FB}K_A + b_t))$. | $0$ (`54,400`) |
| **D. Records & Frontier** | `recordA` | `on` | `on \| off` | *It's Time To Toss The Dice* | **`on`**: forward record $R_A$ updated via $W_{\text{rec\_A}}, W_{\text{cross\_A}}, \text{LN}_{\text{recA}}$. **`off`**: $R_A = 0$; $W_{\text{rec\_A}}, W_{\text{cross\_A}}, W_{\text{cross\_B}}$ bypassed. | $-3,072$ (`51,328`) |
| **D. Records & Frontier** | `recordB` | `on` | `on \| off` | *It's Time To Toss The Dice* | **`on`**: partner record $R_B$ updated via $W_{\text{rec\_B}}, W_{\text{cross\_B}}, \text{LN}_{\text{recB}}$ plus reverse-LM `scoutBackwardPass`. **`off`**: $R_B = 0$ and reverse scout pass disabled. | $-3,072$ (`51,328`) |
| **D. Records & Frontier** | `rbMode` | `causal` | `causal \| prevrev \| legacy` | *It's Time To Toss The Dice* | **`causal`**: $R_B$ carried forward across tokens/chunks. **`prevrev`**: $R_B$ re-seeded at each contiguous chunk boundary from a right-to-left reading of the previous chunk. **`legacy`**: leaky future scout (diagnostic only). | $0$ (`54,400`) |
| **D. Records & Frontier** | `frontierPresent` | `on` | `on \| off` | *It's Time To Toss The Dice* | **`on`**: $\text{logits} = Q(W_{\text{frontier}})R_{\text{comb}} + Q(W_{\text{present}})h_{\text{frontier}}$. **`off`**: record-only frontier ($\text{logits} = Q(W_{\text{frontier}})R_{\text{comb}}$; $W_{\text{present}}$ bypassed). | $-8,192$ (`46,208`) |
| **D. Records & Frontier** | `frontierHelix` | `on` | `on \| off` | *It's Time To Toss The Dice* | **`on`**: $W_{\text{present}}$ reads full helical state $h = \sigma(\rho)\odot G(\theta)h_{\text{prev}} + P$. **`off`**: strips helical recurrence from frontier readout so $W_{\text{present}}$ reads only instantaneous progression $P$. | $0$ (`54,400`) |
| **E. Helical State** | `helix` | `on` | `on \| off` | *The Drill of Entropy* | **`on`**: working state spirals across time via $h = \sigma(\rho)\odot G(\theta)h_{\text{prev}} + P$. **`off`**: disables helical recurrence ($\theta, \rho$ bypassed, $h_{\text{prev}} = 0$, $h = P$). | $-128$ (`54,272`) |
| **E. Helical State** | `helixRotation` | `learned` | `learned \| none` | *The Drill of Entropy* | **`learned`**: pairwise Givens rotations $G(\theta^{(l)})$ applied to $h$. **`none`**: identity rotation $G = I$ (exponential decay without rotation; $\theta^{(l)}$ bypassed). | $-64$ (`54,336`) |
| **E. Helical State** | `helixRetention` | `learned` | `learned \| fixed \| none` | *The Drill of Entropy* | **`learned`**: trainable per-dim decay $\sigma(\rho^{(l)})$ initialized at $\rho = 2.2$. **`fixed`**: constant $\sigma(2.2) \approx 0.90025$ ($\rho$ not stepped). **`none`**: $\sigma \equiv 1$ (v0.0009 integrate-everything behavior). | $-64$ (`54,336`) |
| **F. Thermodynamic Loss** | `thermo` | `hinge` | `off \| linear \| hinge` | *The Past Is Still The Future* | **`hinge`**: $\mathcal{L}_{\text{thermo}} = \lambda \max(0, -\Delta S)$. **`linear`**: $\mathcal{L}_{\text{thermo}} = -\lambda \Delta S$. **`off`**: $\mathcal{L}_{\text{thermo}} = 0$ (no entropy gradient injected into records). | $0$ (`54,400`) |
| **F. Thermodynamic Loss** | `lambda` | `0.05` | `number >= 0` | *The Past Is Still The Future* | Scaling coefficient $\lambda$ on thermodynamic entropy loss $\mathcal{L}_{\text{thermo}}$. Setting `lambda=0` disables thermodynamic loss. | $0$ (`54,400`) |
| **G. Numerics** | `weights` | `ternary` | `ternary \| float32` | *BitNet b1.58 STE* | **`ternary`**: quantizes large matrices to $\{-1, 0, +1\} \times \text{mean}|W|$ in forward/backward via Straight-Through Estimator. **`float32`**: full-precision IEEE-754 float32 weights. | $0$ (`54,400`) |
| **G. Numerics** | `clipMode` | `global` | `global \| per_group` | *Honest Lab Tier 1.6* | **`global`**: clips all parameter tensors jointly by global $L_2$ norm $\le 1.0$. **`per_group`**: clips each parameter tensor individually to $L_2$ norm $\le 1.0$, isolating LayerNorm-beta gradient spikes. | $0$ (`54,400`) |
| **G. Numerics** | `phiClip` | `on` | `on \| off` | *The Electromagnetic Node* | **`on`**: clamps $W_{gA}$ gradients to $\pm \varphi$ and $W_{gB}$ gradients to $\pm 1/\varphi$ after norm clipping. **`off`**: clamps both to $\pm 1.0$. Proven inert under norm clipping $\le 1.0$. | $0$ (`54,400`) |

## Deterministic Settings Strings (`L12:...`)

Every configuration serializes deterministically via `serializeConfig(cfg)` and parses via `parseConfig(str)`:

```text
L12:arch=learns;dim=32;lr=0.032;helix=off;thermo=hinge
```

Both `tests/run.js`, `tests/gradcheck.js`, `tests/optimizer-coverage.js`, and the browser **Switchboard** panel accept either full `L12:...` strings or shorthand `key=value;key2=value2` strings:

```bash
# Verify exact analytical vs finite-difference gradients for any switch:
node tests/gradcheck.js helix=off
node tests/gradcheck.js fpIters=0
node tests/gradcheck.js --all

# Verify optimizer parameter coverage for any switch:
node tests/optimizer-coverage.js relEmbed=off
node tests/optimizer-coverage.js --all

# Run headless training and evaluation with any switch string:
node tests/run.js --corpus corpus.txt --arch learns --config "L12:arch=learns;dim=32;lr=0.032;helix=off" --seed 3 --steps 150 --final
```
