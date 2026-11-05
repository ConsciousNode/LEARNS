# LEARNS v0.0013 — Tier 3 Baselines Reference (`BASELINES.md`)

To evaluate what LEARNS's six paper-derived components contribute on *The Adventures of Sherlock Holmes* (`corpus.txt`, 562,404 ASCII bytes, FNV-1a `CORPUS_HASH = 0x39c84ca9`), LEARNS v0.0013 includes three standard families of baselines inside the **exact same training, chunking, and evaluation harness** (`index.html` and `tests/run.js`):

1. **Interpolated Kneser–Ney $n$-gram tables** ($n \in \{1, 2, 3, 5\}$, both full-split `--ngramData=all` and data-matched `--ngramData=seen`).
2. **Standard 2-layer GRU** (`arch=gru`, matched parameter count, in both `weights=ternary` and `weights=float32`).
3. **2-layer Causal Pre-LN RoPE Transformer** (`arch=transformer`, matched parameter count, in both `weights=ternary` and `weights=float32`).

All neural models share:
- The same deterministic 80% / 10% / 10% contiguous byte split (`splitCorpus80_10_10`: `449,923` train / `56,240` validation / `56,241` test bytes).
- The same chunk stream (`--stream=stride` by default in v0.0013: a deterministic per-epoch seeded permutation of all $7,030$ non-overlapping $64$-byte chunks covering the entire training split once per epoch; or `--stream=head`: the sequential v0.0012 schedule `start = (step * 64) % (N_train - 66)`).
- Identical contiguity handling: whenever consecutive chunks are not adjacent (`start !== prevStart + chunkSize`), the recurrent state (`App.trainState = model.getEmptyState(); App.prevScoutRB = null;`) is reset cleanly before the chunk runs across `LEARNS`, `GRU`, and `Transformer`.
- The same AdamW optimizer (`beta1=0.9, beta2=0.999, eps=1e-8`, weight decay `0.01` on matrix weights, `0.0` on embeddings and biases, `0.1 × lr` on LayerNorm gains/biases), linear warmup (`100` steps), cosine learning-rate decay (`cosineLR`), and L2 gradient norm clipping (`clipGradNorm` at `1.0`).
- The same BitNet b1.58 ternary quantization function (`matvecTernary` / `matvecTernaryT`: $Q(\mathbf{W}) = \text{scale} \cdot \text{round\_clip}(\mathbf{W}/\text{scale}, -1, 1)$ with $\text{scale} = \text{mean}|\mathbf{W}|$ and straight-through estimator) when `weights=ternary`.
- The same causal full-split evaluator (`evalFullSplit`), reporting negative log-likelihood in **nats/byte** and **bits/byte** ($\text{nats} / \ln 2$).

---

## 1. Interpolated Kneser–Ney $n$-Grams (`KneserNeyModel`)

### Equations
For order $M \in \{1, 2, 3, 5\}$:
- **Order 1 (`unigram`, $M=1$)**: Add-one smoothed unigram table over vocabulary $V=256$:
  $$P_{\text{uni}}(w) = \frac{c(w) + 1}{N_{\text{obs}} + V}$$
- **Orders $M \in \{2, 3, 5\}$ (`bigram`, `trigram`, `5gram`)**: Recursive interpolated modified Kneser–Ney smoothing with Ney absolute discounting estimate:
  $$D_m = \frac{n_{1,m}}{n_{1,m} + 2 n_{2,m}} \quad (1 \le m \le M)$$
  where $n_{1,m}$ and $n_{2,m}$ are the counts of $m$-grams occurring exactly once and twice in the count table (highest order $m=M$ uses true token counts; lower orders $1 \le m < M$ use continuation counts $N_{1+}(\cdot, u_{t-m+1}^{t-1}, w) = |\{v : c(v, u_{t-m+1}^{t-1}, w) > 0\}|$). The recursive probability is:
  $$P_{\text{KN}}^{(m)}(w \mid u_{t-m+1}^{t-1}) = \frac{\max(c_m(u_{t-m+1}^{t-1}, w) - D_m, 0)}{c_m(u_{t-m+1}^{t-1}, \cdot)} + \lambda(u_{t-m+1}^{t-1}) P_{\text{KN}}^{(m-1)}(w \mid u_{t-m+2}^{t-1})$$
  $$\lambda(u_{t-m+1}^{t-1}) = \frac{D_m \cdot N_{1+}(u_{t-m+1}^{t-1}, \cdot)}{c_m(u_{t-m+1}^{t-1}, \cdot)}$$
  anchored at the continuation unigram distribution interpolated with uniform $1/V$.

### Full-Split (`--ngramData=all`) vs Data-Matched (`--ngramData=seen`)
- **`--ngramData=all` (default)**: Fits the Kneser–Ney tables across the entire continuous 449,923-byte training split (`KneserNeyModel.fit(train)`).
- **`--ngramData=seen` (v0.0013)**: Reconstructs the exact sequence of chunk start positions walked by the neural models at a given `--steps`, `--stream`, and `--seed`, merges only adjacent consecutive chunks into contiguous segments (`buildSeenSegments`), and fits Kneser–Ney over the exact target bytes $i \in [1, \text{seg.length}-1]$ within each segment (`KneserNeyModel.fitSegments(segments)`) without inventing artificial $n$-gram transitions across non-adjacent chunk boundaries.

| Order | `--ngramData=all` (449,923 B) Test Loss (nats/B) | `--ngramData=seen` 150 steps (`head`, 9,600 B) | `--ngramData=seen` 150 steps (`stride`, 9,600 B) | `--ngramData=seen` 3,515 steps (`stride`, 224,960 B, mean 3 seeds) | `--ngramData=seen` 7,030 steps (`stride`, 449,920 B, 1 pass) |
|---|---:|---:|---:|---:|---:|
| **1-gram (`unigram`)** | `3.0953` (78 entries) | `3.1192` (65) | `3.1174` (58) | `3.0959` (78) | `3.0953` (78) |
| **2-gram (`bigram`)** | `2.4349` (1,568 entries) | `2.5125` (728) | `2.5042` (665) | `2.4368` (1,401) | `2.4349` (1,568) |
| **3-gram (`trigram`)** | `1.9358` (11,042 entries) | `2.1984` (3,416) | `2.1813` (3,189) | `1.9451` (9,289) | `1.9359` (10,985) |
| **5-gram (`5gram`)** | `1.4432` (123,785 entries) | `2.0580` (15,435) | `2.0412` (14,201) | `1.4979` (88,979) | `1.4462` (121,311) |

---

## 2. Parameter-Matched 2-Layer GRU (`GRUModel`)

### Architecture & Equations
For hidden dimension $H = \text{round}(1.53 \times \text{dim})$ ($H = 49$ at $\text{dim}=32$, $H = 98$ at $\text{dim}=64$) and $L=2$ layers:
$$\mathbf{x}_t^{(0)} = \mathbf{W}_{\text{embed}}[x_t] \in \mathbb{R}^H$$
For layer $l \in \{0, \dots, L-1\}$:
$$\mathbf{z}_t^{(l)} = \sigma\left(Q(\mathbf{W}_z^{(l)})[\mathbf{x}_t^{(l)}; \mathbf{h}_{t-1}^{(l)}] + \mathbf{b}_z^{(l)}\right), \qquad \mathbf{r}_t^{(l)} = \sigma\left(Q(\mathbf{W}_r^{(l)})[\mathbf{x}_t^{(l)}; \mathbf{h}_{t-1}^{(l)}] + \mathbf{b}_r^{(l)}\right)$$
$$\tilde{\mathbf{h}}_t^{(l)} = \tanh\left(Q(\mathbf{W}_h^{(l)})[\mathbf{x}_t^{(l)}; \mathbf{r}_t^{(l)} \odot \mathbf{h}_{t-1}^{(l)}] + \mathbf{b}_h^{(l)}\right)$$
$$\mathbf{h}_t^{(l)} = (1 - \mathbf{z}_t^{(l)}) \odot \mathbf{h}_{t-1}^{(l)} + \mathbf{z}_t^{(l)} \odot \tilde{\mathbf{h}}_t^{(l)}, \qquad \mathbf{x}_t^{(l+1)} = \mathbf{h}_t^{(l)}$$
Readout:
$$\text{logits}_t = Q(\mathbf{W}_{\text{out}})\mathbf{h}_t^{(L-1)} + \mathbf{b}_{\text{out}} \in \mathbb{R}^V$$

### Parameter Count Matching (Within 10% of LEARNS)
- At $\text{dim}=32$: LEARNS has `54,400` parameters; GRU ($H=49, L=2$) has $256(49) + 2 \times [3(49 \times 98) + 3(49)] + 256(49) + 256 = \mathbf{54,453}$ parameters (**+0.10%** difference).

---

## 3. Parameter-Matched 2-Layer Pre-LN RoPE Causal Transformer (`TransformerModel`)

### Architecture & Equations
For model width $d = 40$, FFN intermediate width $d_{\text{ff}} = 136$, $n_{\text{heads}} = 4$ ($d_{\text{head}} = 10$), causal window $W = 64$, and $L=2$ layers at $\text{dim}=32$:
$$\mathbf{x}_t^{(0)} = \mathbf{W}_{\text{embed}}[x_t] \in \mathbb{R}^d$$
For each layer $l \in \{0, \dots, L-1\}$:
1. **Pre-LN Multi-Head Causal Attention with RoPE**:
   $$\mathbf{u}_t = \text{LN}_1^{(l)}(\mathbf{x}_t^{(l)}), \quad \mathbf{q}_t = \text{RoPE}(Q(\mathbf{W}_Q^{(l)})\mathbf{u}_t, t), \quad \mathbf{k}_t = \text{RoPE}(Q(\mathbf{W}_K^{(l)})\mathbf{u}_t, t), \quad \mathbf{v}_t = Q(\mathbf{W}_V^{(l)})\mathbf{u}_t$$
   For each head $h \in \{1..4\}$ and causal key positions $s \in [\max(0, t-W+1), t]$:
   $$\alpha_{t,s}^{(h)} = \text{softmax}_s\left(\frac{\langle \mathbf{q}_t^{(h)}, \mathbf{k}_s^{(h)} \rangle}{\sqrt{d_{\text{head}}}}\right), \qquad \mathbf{o}_t^{(h)} = \sum_s \alpha_{t,s}^{(h)} \mathbf{v}_s^{(h)}$$
   $$\mathbf{m}_t^{(l)} = \mathbf{x}_t^{(l)} + Q(\mathbf{W}_O^{(l)})[\mathbf{o}_t^{(1)}; \dots; \mathbf{o}_t^{(4)}]$$
2. **Pre-LN Feed-Forward Block**:
   $$\mathbf{x}_t^{(l+1)} = \mathbf{m}_t^{(l)} + Q(\mathbf{W}_{\text{ff2}}^{(l)})\tanh\left(Q(\mathbf{W}_{\text{ff1}}^{(l)})\text{LN}_2^{(l)}(\mathbf{m}_t^{(l)})\right)$$
3. **Final LayerNorm & Readout**:
   $$\text{logits}_t = Q(\mathbf{W}_{\text{out}})\text{LN}_f(\mathbf{x}_t^{(L)}) \in \mathbb{R}^V$$

### Parameter Count Matching (Within 10% of LEARNS)
- At $\text{dim}=32$ ($d=40, d_{\text{ff}}=136, L=2$): $256(40) + 2 \times [4(40^2) + 2(40 \times 136) + 4(40)] + 2(40) + 256(40) = \mathbf{55,440}$ parameters (**+1.91%** difference vs LEARNS `54,400`).

---

## 4. Validation Learning-Rate Re-Sweep (`--stream=stride`, 1,000 Steps, Seed 3)

In v0.0013, every neural contender's learning rate was re-swept across 3 values around the v0.0012 winner at 1,000 steps under `--stream=stride` on **validation only** (`seed=3`, test split untouched):

| Architecture (`dim=32`, `--stream=stride`, 1,000 steps, `seed=3`) | `lr = 0.004` | `lr = 0.008` | `lr = 0.016` | `lr = 0.032` | Selected Winner (Val Loss) |
|---|---:|---:|---:|---:|---|
| **LEARNS default (`weights=ternary`)** | `2.599483` | `2.438120` | **`2.316919`** | — | **`lr = 0.016`** (`2.316919`) |
| **LEARNS `relEmbed=off` (`weights=ternary`)** | `2.578271` | `2.424215` | **`2.309079`** | — | **`lr = 0.016`** (`2.309079`) |
| **Transformer (`weights=ternary`)** | `2.361972` | `2.321222` | **`2.318383`** | — | **`lr = 0.016`** (`2.318383`) |
| **Transformer (`weights=float32`)** | `2.324468` | **`2.309706`** | `2.462100` | — | **`lr = 0.008`** (`2.309706`) |
| **GRU (`weights=ternary`)** | — | `2.081710` | **`2.058522`** | `2.207049` | **`lr = 0.016`** (`2.058522`) |
| **GRU (`weights=float32`)** | — | **`1.956874`** | `1.985888` | `2.131414` | **`lr = 0.008`** (`1.956874`) |
