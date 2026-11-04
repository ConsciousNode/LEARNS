# LEARNS v0.0012 — Baselines & Fairness Specification (`BASELINES.md`)

To measure what LEARNS's six paper-derived components actually contribute, LEARNS v0.0012 embeds three baseline model families inside the same marked slice (`/*@@LEARNS_BEGIN*/` ... `/*@@LEARNS_END*/`, `CODE_HASH: 0xfe2cf657`):

1. **Interpolated Kneser–Ney $n$-gram tables** (`unigram`, `bigram`, `trigram`, `5gram`)
2. **Standard Multi-Layer GRU** (`gru`, in both `weights=ternary` and `weights=float32`)
3. **Tiny Pre-LayerNorm Causal RoPE Transformer** (`transformer`, in both `weights=ternary` and `weights=float32`)

All architectures share the exact same byte-level (or BPE) tokenizer, contiguous 80% / 10% / 10% train/validation/test split (`splitCorpus80_10_10`), seeded PRNG (`mulberry32`), contiguous chunk stream (`chunk = 64`), BPTT length (`64`), AdamW optimizer (`beta1 = 0.9, beta2 = 0.999, eps = 1e-8, wd = 0.01`, with `wd = 0` on token embeddings), cosine learning-rate schedule, and causal full-split evaluator (`evalFullSplit`).

## 1. Interpolated Kneser–Ney $n$-Gram Floor (`KneserNeyModel`)

Fitted exclusively on the 80% training split (`449,923` bytes of `corpus.txt`, `CORPUS_HASH: 0x39c84ca9`):

- **Unigram ($n = 1$)**: Laplace-smoothed empirical byte distribution $P_1(w) = \frac{c(w) + 1}{T_{\text{train}} + V}$.
- **Higher-order Interpolated Kneser–Ney ($n \in \{2, 3, 5\}$)**: Following Chen & Goodman (1999), the highest order $m = n$ uses raw training $n$-gram counts $a_n(u w) = c(u w)$, while lower orders $1 \le m < n$ use continuation counts $a_m(u w) = N_{1+}(\cdot u w) = |\{v : c(v u w) > 0\}|$. At each order $m$, the absolute discount $D_m = \frac{n_1^{(m)}}{n_1^{(m)} + 2 n_2^{(m)}}$ is estimated from training singleton and doubleton counts $n_1^{(m)}, n_2^{(m)}$, and probabilities are recursively interpolated:
  $$P_{\text{KN}}^{(m)}(w \mid u) = \frac{\max(a_m(u w) - D_m, 0)}{a_m(u \cdot)} + \frac{D_m \, N_{1+}(u \cdot)}{a_m(u \cdot)} P_{\text{KN}}^{(m-1)}(w \mid \text{suffix}(u))$$
  with unigram continuation base $P_{\text{KN}}^{(0)}(w) = 1/V$.

| Baseline (`arch`) | Order $n$ | Non-Zero Table Entries | Validation Loss (nats/byte) | Validation bits/byte | Test Loss (nats/byte) | Test bits/byte |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `unigram` | 1 | 78 | 3.078542 | 4.441397 | 3.095295 | 4.465567 |
| `bigram` | 2 | 1,568 | 2.403409 | 3.467386 | 2.434862 | 3.512763 |
| `trigram` | 3 | 11,042 | 1.900319 | 2.741581 | 1.935779 | 2.792739 |
| `5gram` | 5 | 123,785 | 1.406447 | 2.029074 | 1.443163 | 2.082045 |

## 2. Standard GRU Language Model (`GRUModel`)

A 2-layer Gated Recurrent Unit (Cho et al., 2014) language model with hidden dimension $H = \text{round}(1.53 \times \text{dim})$ chosen to match LEARNS's parameter count within $\pm 1.2\%$:

- **Embedding**: $x_t^{(0)} = W_{\text{embed}}[w_t] \in \mathbb{R}^H$, initialized $\mathcal{N}(0, 1)$, weight decay $0$.
- **Recurrent Layers ($l = 0, \dots, L-1$)**: Given layer input $x_t^{(l)} \in \mathbb{R}^H$ and carried hidden state $h_{t-1}^{(l)} \in \mathbb{R}^H$:
  $$\begin{aligned}
  z_t^{(l)} &= \sigma\!\left(\mathcal{M}(W_z^{(l)})\,[x_t^{(l)}, h_{t-1}^{(l)}] + b_z^{(l)}\right) \\
  r_t^{(l)} &= \sigma\!\left(\mathcal{M}(W_r^{(l)})\,[x_t^{(l)}, h_{t-1}^{(l)}] + b_r^{(l)}\right) \\
  \tilde{h}_t^{(l)} &= \tanh\!\left(\mathcal{M}(W_h^{(l)})\,[x_t^{(l)}, r_t^{(l)} \odot h_{t-1}^{(l)}] + b_h^{(l)}\right) \\
  h_t^{(l)} &= (1 - z_t^{(l)}) \odot h_{t-1}^{(l)} + z_t^{(l)} \odot \tilde{h}_t^{(l)}
  \end{aligned}$$
  where $\mathcal{M}(W)$ is `matvecTernary` when `weights=ternary` (BitNet b1.58 STE) and full-precision `matvec` when `weights=float32`. Weight matrices are initialized with Xavier scale $1/\sqrt{H}$.
- **Readout**: $\text{logits}_t = \mathcal{M}(W_{\text{out}})\,h_t^{(L-1)} + b_{\text{out}} \in \mathbb{R}^V$.
- **State Continuity**: Hidden states $\{h_t^{(l)}\}_{l=0}^{L-1}$ are carried across contiguous chunks during both training and full-split evaluation, with exact BPTT across each 64-token chunk.

## 3. Tiny Pre-LayerNorm Causal RoPE Transformer (`TransformerModel`)

A 2-layer decoder-only causal Transformer (Vaswani et al., 2017; Su et al., 2021) with $n_{\text{heads}} = 4$ and causal attention window equal to LEARNS's chunk context length (`chunk = 64`):

- **Embedding**: $x_t^{(0)} = W_{\text{embed}}[w_t] \in \mathbb{R}^d$, initialized $\mathcal{N}(0, 1)$, weight decay $0$.
- **Pre-LN Causal Multi-Head Self-Attention ($l = 0, \dots, L-1$)**:
  - $u_t = \text{LayerNorm}(x_t^{(l)}; \gamma_{1}^{(l)}, \beta_{1}^{(l)})$
  - $q_t = \text{RoPE}(\mathcal{M}(W_Q^{(l)}) u_t, t), \quad k_t = \text{RoPE}(\mathcal{M}(W_K^{(l)}) u_t, t), \quad v_t = \mathcal{M}(W_V^{(l)}) u_t$
  - For each head $h \in \{1, \dots, 4\}$ of dimension $d_h = d / 4$, causal attention is computed over the window $s \in \{\max(0, t - W_{\text{ctx}} + 1), \dots, t\}$:
    $$\alpha_{t,s}^{(h)} = \text{softmax}_{s \le t}\!\left(\frac{\langle q_t^{(h)}, k_s^{(h)} \rangle}{\sqrt{d_h}}\right), \qquad a_t^{(h)} = \sum_{s \le t} \alpha_{t,s}^{(h)} v_s^{(h)}$$
  - Residual connection: $m_t^{(l)} = x_t^{(l)} + \mathcal{M}(W_O^{(l)})\,\text{concat}(a_t^{(1)}, \dots, a_t^{(4)})$
- **Pre-LN Feed-Forward Block ($l = 0, \dots, L-1$)**:
  - $n_t = \text{LayerNorm}(m_t^{(l)}; \gamma_{2}^{(l)}, \beta_{2}^{(l)})$
  - $x_t^{(l+1)} = m_t^{(l)} + \mathcal{M}(W_{\text{ff2}}^{(l)})\,\tanh\!\left(\mathcal{M}(W_{\text{ff1}}^{(l)})\,n_t\right)$ where $W_{\text{ff1}}^{(l)} \in \mathbb{R}^{d_{\text{ff}} \times d}, W_{\text{ff2}}^{(l)} \in \mathbb{R}^{d \times d_{\text{ff}}}$.
- **Final Readout**: $\text{logits}_t = \mathcal{M}(W_{\text{out}})\,\text{LayerNorm}(x_t^{(L)}; \gamma_f, \beta_f)$.
- **Exact Window BPTT**: During backward, gradients propagate through all queries, keys, and values across the entire causal window (`kv` cache), verified to match finite differences at $\cos = 1.000000$ across all 24 parameter tensors.

## 4. Matched Parameter Counts & Fairness Verification

| Scale | Architecture (`arch`) | Internal Dimensions ($L = 2, V = 256$) | Trainable Parameters | Relative Diff vs LEARNS | Within $\pm 10\%$ Rule? | Gradcheck (`gradcheck.js`) | Optimizer Coverage (`optimizer-coverage.js`) |
| :--- | :--- | :--- | ---: | ---: | :--- | :--- | :--- |
| **Scale 1 (`dim=32`)** | `learns` | `dim = 32, K_hist = 32` | **54,400** (`54,880` w/ LN) | `0.00%` | Yes (Reference) | `ALL OK` ($\cos = 1.000000$) | `ALL OK` (28/28 stepped) |
| **Scale 1 (`dim=32`)** | `gru` | `H = 49` | **54,450** | `+0.09%` (`-0.78%` w/ LN) | **Yes** ($\pm 0.8\%$) | `ALL OK` ($\cos = 1.000000$) | `ALL OK` (9/9 stepped) |
| **Scale 1 (`dim=32`)** | `transformer` | `d = 40, d_ff = 136, nHeads = 4` | **55,440** | `+1.91%` (`+1.02%` w/ LN) | **Yes** ($\pm 1.9\%$) | `ALL OK` ($\cos = 1.000000$) | `ALL OK` (14/14 stepped) |
| **Scale 2 (`dim=64`)** | `learns` | `dim = 64, K_hist = 32` | **168,192** (`169,152` w/ LN) | `0.00%` | Yes (Reference) | `ALL OK` ($\cos = 1.000000$) | `ALL OK` (28/28 stepped) |
| **Scale 2 (`dim=64`)** | `gru` | `H = 98` | **166,268** | `-1.14%` (`-1.70%` w/ LN) | **Yes** ($\pm 1.7\%$) | `ALL OK` ($\cos = 1.000000$) | `ALL OK` (9/9 stepped) |
| **Scale 2 (`dim=64`)** | `transformer` | `d = 72, d_ff = 310, nHeads = 4` | **168,336** | `+0.09%` (`-0.48%` w/ LN) | **Yes** ($\pm 0.5\%$) | `ALL OK` ($\cos = 1.000000$) | `ALL OK` (14/14 stepped) |

## 5. Fairness Rules Enforced

1. **Identical Data & Splits**: Every model sees the exact same byte stream from `corpus.txt` (`0x39c84ca9`), split 80% train (`449,923` bytes), 10% validation (`56,240` bytes), and 10% test (`56,241` bytes).
2. **Identical Tuning Protocol**: Every neural model variant (`learns`, `gru`, `transformer` $\times$ `ternary`, `float32` $\times$ `dim=32`, `dim=64`) undergoes the exact same 5-point log-spaced learning-rate sweep (`lr ∈ {0.002, 0.004, 0.008, 0.016, 0.032}`) on the validation split (`seed = 3, steps = 150`), and the validation-winning learning rate is locked before any `--final` test evaluation is run.
3. **Identical Seeds & Step Budget**: All final comparisons use the exact same 5 seeds (`3, 7, 11, 19, 23`), chunk size (`64`), warmup (`100`), and step budget (`150`).
4. **Verified Gradients & Optimizer Coverage**: Both `node tests/gradcheck.js --all` and `node tests/optimizer-coverage.js --all` pass with zero errors across all baselines.
