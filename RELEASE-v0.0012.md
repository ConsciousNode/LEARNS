# LEARNS v0.0012 — The Honest Lab

**Logical Entropic Arrow Recursive Neuro-Symbolic Architecture**  
*ConsciousNode SoftWorks — Single-File Browser & Node Language Model Lab*

- **Code Stamp (`CODE_HASH`)**: `0xfe2cf657`
- **Corpus Stamp (`CORPUS_HASH`)**: `0x39c84ca9` (*The Adventures of Sherlock Holmes*, 562,404 ASCII bytes)

## Where It Stands (v0.0012 Measured Results)

In v0.0012 (*The Honest Lab*), we replaced the 300-byte held-out snippet with a deterministic, contiguous **80% train (`449,923` bytes) / 10% validation (`56,240` bytes) / 10% test (`56,241` bytes)** split of *The Adventures of Sherlock Holmes* (`corpus.txt`), turned all six paper-derived components into switches with exact analytical gradients (`node tests/gradcheck.js --all` $\to$ `ALL OK` across 31 configurations; `node tests/optimizer-coverage.js --all` $\to$ `ALL OK` across 32 configurations), and added parameter-matched baselines (Interpolated Kneser–Ney $n$-grams, a 2-layer GRU, and a 2-layer Pre-LN RoPE Causal Transformer, in both BitNet b1.58 ternary STE and float32) to the exact same harness.

### Baseline Leaderboard on the 56 KB Test Split (`steps = 150`, 5 seeds: `3, 7, 11, 19, 23`)

| Model | Params (`dim=32` / `dim=64` scale) | Mean Test Loss (nats/byte) | Mean Test bits/byte | Min–Max Across 5 Seeds | Throughput (tok/s) |
| :--- | ---: | ---: | ---: | :--- | ---: |
| **5-gram Kneser–Ney** | 123,785 | **1.4432** | **2.0820** | `1.4432` | 46,866 |
| **Trigram Kneser–Ney** | 11,042 | **1.9358** | **2.7927** | `1.9358` | 323,213 |
| **GRU (`dim=64, float32`)** | 166,268 | **2.3981** | **3.4597** | `[2.3696, 2.4183]` | 1,362 |
| **Bigram Kneser–Ney** | 1,568 | **2.4349** | **3.5128** | `2.4349` | 379,993 |
| **GRU (`dim=32, float32`)** | 54,450 | **2.5211** | **3.6372** | `[2.4777, 2.5623]` | 3,539 |
| **Transformer (`dim=64, float32`)** | 168,336 | **2.7819** | **4.0134** | `[2.7575, 2.8017]` | 923 |
| **Transformer (`dim=64, ternary`)** | 168,336 | **2.8138** | **4.0594** | `[2.7856, 2.8554]` | 865 |
| **GRU (`dim=32, ternary`)** | 54,450 | **2.8348** | **4.0898** | `[2.6708, 2.9320]` | 3,274 |
| **LEARNS (`dim=32, ternary, relEmbed=off`)** | 53,376 | **2.8498** | **4.1114** | `[2.7458, 2.9245]` | 456 |
| **Transformer (`dim=32, float32`)** | 55,440 | **2.8591** | **4.1248** | `[2.8216, 2.9048]` | 1,685 |
| **Transformer (`dim=32, ternary`)** | 55,440 | **2.8943** | **4.1756** | `[2.8439, 2.9380]` | 1,617 |
| **GRU (`dim=64, ternary`)** | 166,268 | **2.9204** | **4.2132** | `[2.7490, 3.0305]` | 1,300 |
| **LEARNS (`dim=64, float32`)** | 168,192 | **2.9339** | **4.2327** | `[2.7403, 3.1362]` | 184 |
| **LEARNS (`dim=32, ternary, full default`)** | 54,400 | **3.0221** | **4.3600** | `[2.9285, 3.1107]` | 550 |
| **Unigram Kneser–Ney** | 78 | **3.0953** | **4.4656** | `3.0953` | 1,519,973 |
| **LEARNS (`dim=32, float32`)** | 54,400 | **3.1289** | **4.5141** | `[3.1275, 3.1300]` | 530 |
| **LEARNS (`dim=64, ternary`)** | 168,192 | **3.1419** | **4.5329** | `[3.1253, 3.1573]` | 246 |

*(At 600 steps with `lr=0.008`, `LEARNS dim=32 ternary` reaches `2.5311` nats/byte on validation, still trailing a 150-step `GRU dim=32 float32` at `2.4972`.)*

### What the Ablations Showed

1. **What works in LEARNS**: The damped helical state (`helix=on`, `helixRetention=learned`) projected to the frontier via `W_present` (`frontierPresent=on`, `frontierHelix=on`). Turning any of these off hurts on all 5/5 seeds (`+0.136` to `+0.746` nats/byte worse).
2. **What actively hurts**: Relational Embedding (`relEmbed=on`). Turning it off (`relEmbed=off`) improves test loss on all 5/5 seeds by **$-0.1723$ nats/byte** (`3.0221 -> 2.8498`).
3. **What is noise or non-convergent**:
   - **Dyadic Fixed Point (`fpIters`)**: Because `LayerNorm` inside the Jacobi loop resets the vector norm to $\sqrt{{d}}$ at every step, the iteration never converges (mean final residual `6.37` at 6 iterations, `3.85` at 12 iterations). Setting `fpIters=0` (single feed-forward pass) is statistically indistinguishable from 6 iterations (`+0.0488` nats, crosses zero) and 18% faster; enabling input injection (`fpInject=on`) lowers the residual (`1.46`) but hurts test loss on 5/5 seeds (`+0.7919` nats).
   - **Opposite Arrows (`rbMode`)**: The skew-symmetric rotation cosine between $W_{{\text{{rec\_A}}}}$ and $W_{{\text{{rec\_B}}}}$ averages `-0.017` in `causal` and `+0.029` in `prevrev` (both spanning negative and positive values across seeds), with no significant loss difference (`+0.0245` nats, noise).
   - **$\varphi$-Clip (`phiClip`)**: Confirmed 100% inert (`0.000000` difference on every seed) because post-clip gradient elements never reach $1/\varphi \approx 0.618$.
   - **LayerNorm-Beta Gradient Spikes**: Caused by `invStd = 316.23` compounding as $(316.23 \cdot W)^k$ across unrolled fixed-point iterations when `suffix_sig = 0` or `h_prev = 0` at empty states (occurring on 2.5% of steps, up to $8.26 \times 10^8$). Isolating clips via `clipMode=per_group` improves mean loss by `-0.0213` nats (noise over 150 steps).

## Quick Start & Reproducibility

```bash
# 1. Verify exact analytical gradients across all 31 switch and baseline configurations:
node tests/gradcheck.js --all

# 2. Verify optimizer parameter coverage across all 32 switch and baseline configurations:
node tests/optimizer-coverage.js --all

# 3. Run any experiment headlessly (prints one stamped JSON line):
node tests/run.js --corpus corpus.txt --arch learns --config "dim=32;lr=0.032" --seed 3 --steps 150 --final
node tests/run.js --corpus corpus.txt --arch gru --config "dim=64;weights=float32;lr=0.008" --seed 3 --steps 150 --final
```
