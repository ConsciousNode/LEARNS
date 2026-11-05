# LEARNS v0.0014 ("Signal and Noise") — Tier 3 Baselines Reference (`BASELINES.md`)

To evaluate what LEARNS's six paper-derived components contribute on *The Adventures of Sherlock Holmes* (`corpus.txt`, 562,404 ASCII bytes, FNV-1a `CORPUS_HASH = 0x39c84ca9`), LEARNS v0.0014 includes three standard families of baselines inside the **exact same training, chunking, and evaluation harness** (`index.html` and `tests/run.js`):

1. **Interpolated Kneser–Ney $n$-gram tables** ($n \in \{1, 2, 3, 5\}$, both full-split `--ngramData=all` and data-matched `--ngramData=seen`).
2. **Standard 2-layer GRU** (`arch=gru`, matched parameter count and state capacity at both `dim=32` and `dim=64`, in `weights=ternary` and `weights=float32`).
3. **2-layer Causal Pre-LN RoPE Transformer** (`arch=transformer`, matched parameter count, in `weights=ternary` and `weights=float32`).

All neural models share:
- The same deterministic 80% / 10% / 10% contiguous byte split (`splitCorpus80_10_10`: `449,923` train / `56,240` validation / `56,241` test bytes).
- The same chunk stream (`--stream=stride` by default: a deterministic per-epoch seeded permutation of all $7,030$ non-overlapping $64$-byte chunks covering the entire training split once per epoch; `--stream=block`: one deterministic random contiguous block of `steps * 64` bytes per seed; or `--stream=head`: the sequential v0.0011/v0.0012 schedule `start = (step * 64) % (N_train - 66)`).
- Identical contiguity handling: whenever consecutive chunks are not adjacent (`start !== prevStart + chunkSize`), the recurrent state (`App.trainState = model.getEmptyState(); App.prevScoutRB = null;`) is reset cleanly before the chunk runs across `LEARNS`, `GRU`, and `Transformer`.
- The same AdamW optimizer (`beta1=0.9, beta2=0.999, eps=1e-8`, weight decay `0.01` on matrix weights, `0.0` on embeddings and biases, `0.1 × lr` on LayerNorm gains/biases), linear warmup (`100` steps), cosine learning-rate decay (`cosineLR`), and L2 gradient norm clipping (`clipGradNorm` at `1.0`).
- The same BitNet b1.58 ternary quantization function (`matvecTernary` / `matvecTernaryT`: $Q(\mathbf{W}) = \text{scale} \cdot \text{round\_clip}(\mathbf{W}/\text{scale}, -1, 1)$ with $\text{scale} = \text{mean}|\mathbf{W}|$ and straight-through estimator) when `weights=ternary`.
- The same causal full-split evaluator (`evalFullSplit`) and 5-checkpoint tail jitter band (`--tail 5`, evaluating full validation loss at steps `S-40, S-30, S-20, S-10, S`).

---

## 1. Interpolated Kneser–Ney $n$-Grams (`KneserNeyModel`)

### Equations
For order $M \in \{1, 2, 3, 5\}$:
- **Order 1 (`unigram`, $M=1$)**: Add-one smoothed unigram table over vocabulary $V=256$:
  $$P_{\text{uni}}(w) = \frac{c(w) + 1}{N_{\text{obs}} + V}$$
- **Orders $M \in \{2, 3, 5\}$ (`bigram`, `trigram`, `5gram`)**: Recursive interpolated modified Kneser–Ney smoothing with Ney absolute discounting estimate:
  $$D_m = \frac{n_{1,m}}{n_{1,m} + 2 n_{2,m}} \quad (1 \le m \le M)$$
  $$P_{\text{KN}}^{(m)}(w \mid u_{t-m+1}^{t-1}) = \frac{\max(c_m(u_{t-m+1}^{t-1}, w) - D_m, 0)}{c_m(u_{t-m+1}^{t-1}, \cdot)} + \lambda(u_{t-m+1}^{t-1}) P_{\text{KN}}^{(m-1)}(w \mid u_{t-m+2}^{t-1})$$
  $$\lambda(u_{t-m+1}^{t-1}) = \frac{D_m \cdot N_{1+}(u_{t-m+1}^{t-1}, \cdot)}{c_m(u_{t-m+1}^{t-1}, \cdot)}$$

### Full-Split (`--ngramData=all`) vs Data-Matched (`--ngramData=seen`)

| Order | `--ngramData=all` (449,923 B) Test Loss (nats/B) | `--ngramData=seen` 150 steps (`stride`, 9,600 B) | `--ngramData=seen` 1,000 steps (`stride`, 64,000 B) | `--ngramData=seen` 7,030 steps (`stride`, 449,920 B, 1 pass) |
|---|---:|---:|---:|---:|
| **1-gram (`unigram`)** | `3.0953` | `3.1174` | `3.0988` | `3.0953` |
| **2-gram (`bigram`)** | `2.4349` | `2.5042` | `2.4419` | `2.4349` |
| **3-gram (`trigram`)** | `1.9358` | `2.1813` | `1.9805` | `1.9359` |
| **5-gram (`5gram`)** | `1.4432` | `2.0412` | `1.6492` | `1.4462` |

---

## 2. Parameter-Matched 2-Layer GRU (`GRUModel`)

### Architecture & Equations
For hidden dimension $H = 49$ (at $\text{dim}=32$) or $H = 96$ (at $\text{dim}=64$) and $L=2$ layers:
$$\mathbf{x}_t^{(0)} = \mathbf{W}_{\text{embed}}[x_t] \in \mathbb{R}^H$$
For layer $l \in \{0, \dots, L-1\}$:
$$\mathbf{z}_t^{(l)} = \sigma\left(Q(\mathbf{W}_z^{(l)})[\mathbf{x}_t^{(l)}; \mathbf{h}_{t-1}^{(l)}] + \mathbf{b}_z^{(l)}\right), \qquad \mathbf{r}_t^{(l)} = \sigma\left(Q(\mathbf{W}_r^{(l)})[\mathbf{x}_t^{(l)}; \mathbf{h}_{t-1}^{(l)}] + \mathbf{b}_r^{(l)}\right)$$
$$\tilde{\mathbf{h}}_t^{(l)} = \tanh\left(Q(\mathbf{W}_h^{(l)})[\mathbf{x}_t^{(l)}; \mathbf{r}_t^{(l)} \odot \mathbf{h}_{t-1}^{(l)}] + \mathbf{b}_h^{(l)}\right)$$
$$\mathbf{h}_t^{(l)} = (1 - \mathbf{z}_t^{(l)}) \odot \mathbf{h}_{t-1}^{(l)} + \mathbf{z}_t^{(l)} \odot \tilde{\mathbf{h}}_t^{(l)}, \qquad \mathbf{x}_t^{(l+1)} = \mathbf{h}_t^{(l)}$$
Readout:
$$\text{logits}_t = Q(\mathbf{W}_{\text{out}})\mathbf{h}_t^{(L-1)} + \mathbf{b}_{\text{out}} \in \mathbb{R}^V$$

### Parameter Count & State Capacity Matching (`dim=32` and `dim=64`)

| Scale | Architecture | Active Parameters | Parameter Delta vs LEARNS | Recurrent State Floats Carried Across Tokens |
|---|---|---:|---:|---|
| **`dim=32`** | **LEARNS (`shared`)** | `54,400` | — | **`96` floats** ($\mathbf{R}_A \in \mathbb{R}^{32}, \mathbf{R}_B \in \mathbb{R}^{32}, \mathbf{h} \in \mathbb{R}^{32}$) |
| **`dim=32`** | **LEARNS (`per_layer`)** | `54,400` | `0.00%` | **`128` floats** ($\mathbf{R}_A \in \mathbb{R}^{32}, \mathbf{R}_B \in \mathbb{R}^{32}, \mathbf{h}^{(0)} \in \mathbb{R}^{32}, \mathbf{h}^{(1)} \in \mathbb{R}^{32}$) |
| **`dim=32`** | **GRU ($H=49, L=2$)** | `54,450` | **`+0.09%`** | **`98` floats** ($\mathbf{h}^{(0)} \in \mathbb{R}^{49}, \mathbf{h}^{(1)} \in \mathbb{R}^{49}$) |
| **`dim=64`** | **LEARNS (`shared`)** | `164,224` | — | **`192` floats** ($\mathbf{R}_A \in \mathbb{R}^{64}, \mathbf{R}_B \in \mathbb{R}^{64}, \mathbf{h} \in \mathbb{R}^{64}$) |
| **`dim=64`** | **GRU ($H=96, L=2$)** | `165,914` | **`+1.03%`** | **`192` floats** ($\mathbf{h}^{(0)} \in \mathbb{R}^{96}, \mathbf{h}^{(1)} \in \mathbb{R}^{96}$) |

---

## 3. Parameter-Matched 2-Layer Pre-LN RoPE Causal Transformer (`TransformerModel`)

### Architecture & Equations
At $\text{dim}=32$ ($d = 40$, $d_{\text{ff}} = 136$, $n_{\text{heads}} = 4$, causal window $W = 64$, $L=2$ layers):
- Active parameters: **`54,656`** (**`+0.47%`** vs LEARNS `54,400`).

---

## 4. Validation Learning-Rate Sweeps (`--stream=stride`, 1,000 Steps, Seed 3)

Every neural configuration's learning rate was selected strictly on the validation split (`seed=3`, `1,000` steps, `--stream=stride`, test split untouched):

| Model Family | Candidate `lr = 0.002` | Candidate `lr = 0.004` | Candidate `lr = 0.008` | Candidate `lr = 0.016` | Candidate `lr = 0.024 / 0.032` | Selected Winner (Val Loss) |
|---|---:|---:|---:|---:|---:|---|
| **LEARNS `weights=ternary` (`dim=32`)** | — | `2.5995` | `2.4381` | **`2.3220`** | — | **`lr = 0.016`** (`2.3220`) |
| **LEARNS `weights=float32` (`dim=32`)** | `2.7055` | `4.5793` | `2.2166` | **`2.2142`** | — | **`lr = 0.016`** (`2.2142`) |
| **LEARNS `scoutScale=0` (`dim=32, ternary`)** | — | — | `2.2434` | **`2.1764`** | `2.2343` (`0.024`) | **`lr = 0.016`** (`2.1764`) |
| **LEARNS `dim=64` (`weights=ternary`)** | — | `2.2831` | **`2.1644`** | `2.2105` | — | **`lr = 0.008`** (`2.1644`) |
| **GRU `dim=64` (`weights=ternary`)** | — | `1.9722` | **`1.9326`** | `2.0672` | — | **`lr = 0.008`** (`1.9326`) |
| **GRU `dim=32` (`weights=ternary`)** | — | — | `2.0817` | **`2.0585`** | `2.2070` (`0.032`) | **`lr = 0.016`** (`2.0585`) |
| **GRU `dim=32` (`weights=float32`)** | — | — | **`1.9569`** | `1.9859` | `2.1314` (`0.032`) | **`lr = 0.008 / 0.016`** (`1.9569 / 1.9859`) |
| **Transformer `dim=32` (`weights=ternary`)** | — | `2.3620` | `2.3212` | **`2.3184`** | — | **`lr = 0.016`** (`2.3184`) |
| **Transformer `dim=32` (`weights=float32`)** | — | `2.3245` | **`2.3097`** | `2.4621` | — | **`lr = 0.008`** (`2.3097`) |
