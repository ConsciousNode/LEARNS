# LEARNS

![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/platform-browser-blue?style=flat-square)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen?style=flat-square)
![Weights](https://img.shields.io/badge/weights-ternary%20b1.58-8b5cf6?style=flat-square)
![Hardware](https://img.shields.io/badge/hardware-any%20CPU-a78bfa?style=flat-square)
![Version](https://img.shields.io/badge/version-0.0011-6d28d9?style=flat-square)

**Logical Entropic Arrow Recursive Neuro-Symbolic Architecture.**
**A language model built from scratch, from a physics hypothesis, that trains in a browser tab on whatever CPU you have.**

[**→ Try it live**](https://consciousnode.github.io/LEARNS) · [ConsciousNode SoftWorks](https://consciousnode.github.io)

---

## What it is

LEARNS is ConsciousNode SoftWorks' fully in-house machine-learning architecture. It isn't a transformer, it isn't RWKV, and it isn't a fine-tune of anything. It's a new recurrent model whose six components were each derived from one of the studio's theory papers. Every line, from the BPE tokenizer to the AdamW optimizer, lives in one HTML file. There are no dependencies, no server, no GPU and no install.

The central idea: a mind keeps **two records of time**. R_A carries the past forward; R_B is meant to carry the arrow the other way. The two talk to each other through a fixed-point "now," and a thermodynamic term keeps the records from collapsing into dead order.

**Status: experimental, and honestly so.** As of v0.0010, LEARNS learns: it reads its input and beats the unigram floor. It isn't competitive with established architectures, and isn't trying to be yet. The question it exists to ask is whether these components, taken from physics, add up to something that learns. As of tonight the answer has moved from "unknown" to "yes, it learns." Whether it does something *transformers can't* is still open, and you're invited to help find out.

---

## The six components

| # | Component | What it does | Derived from |
|---|---|---|---|
| 1 | **Relational Embedding** | A token's meaning is its embedding *plus* a projection of the current records: meaning is relational | *It's All Relative* |
| 2 | **Temporal Turing Core** | Component A runs an inner fixed point over a suffix signal of recent tokens; Component B reads the helix and the records | *The Electromagnetic Node* |
| 3 | **Dyadic Fixed Point** | K_A and K_B iterate against each other (K_A = F_A(K_B), K_B = F_B(K_A)) until they agree: the "now" as a fixed point | *K-H Fixed Point Theorem* |
| 4 | **Record / Frontier split** | Two records (R_A, R_B) accumulate what has happened; the frontier reads them (and, since v0.0010, the helix) to guess what happens next | *It's Time To Toss The Dice* |
| 5 | **Helical State** | The working state rotates through learned Givens angles and now decays gently (a damped spiral) as new input is mixed in | *The Drill of Entropy* |
| 6 | **Thermodynamic Loss** | Penalizes the records for *losing* entropy. Holding or gaining it is free, so they can stay structured rather than drift to noise | *The Past Is Still The Future* |

All big matrices are **ternary** (BitNet b1.58: every weight is −1, 0 or +1 times one scale per matrix). Training uses the exact straight-through gradient, true backpropagation through time across the records and the helix, and backprop through the unrolled fixed-point iterations.

---

## Try it

→ **[consciousnode.github.io/LEARNS](https://consciousnode.github.io/LEARNS)**

Or download [`index.html`](index.html) and open it locally. It works fully offline.

1. **Load a `.txt` corpus** and press **COMPILE BPE**.
2. Leave the defaults on the first run: LR 4e-3, thermo mode *hinge*, "frontier reads the present" on, R_B mode *causal*.
3. Press **START TRAINING**. Watch held-out validation loss against the unigram baseline shown beside it. Beating that baseline means the model is using context.
4. **GENERATE** or **CHAT** with what you've trained. **EXPORT .learns** saves the model, optimizer state and settings; import it to keep training later.

Smaller models train faster: dim 64 with 2 layers is a good first run on a laptop. The defaults (dim 128) are fine on anything from the last decade.

---

## Where it stands (v0.0010)

Byte-level, 2 layers, 46 KB English corpus. Loss is measured causally on a held-out tail the model never trains on, so it can't peek at the answer:

| | held-out loss (nats/byte) |
|---|---|
| uniform guess | 5.55 |
| v0.0009, 600 steps | 3.52 (flat) |
| unigram baseline | 3.47 |
| **v0.0010, dim 128, 300 steps** | **3.23** |
| **v0.0010, dim 32, 1000–1500 steps** | **2.90 – 3.20** (across seeds and modes) |
| bigram baseline (add-one smoothed) | 2.80 |

After training, changing the *current* token moves the model's prediction by about 1 nat. Before v0.0010 it moved it by 0.0000000004, which is to say the model wasn't reading its input at all. The [changelog](#changelog) has the full story.

### Open questions (help wanted)

- **Do R_A and R_B learn opposite arrows?** The papers predict opposite chirality. In leak-free *causal* mode, R_B runs forward too, so nothing asks it to be the backward arrow, and we measure no reliable divergence. The `prevrev` R_B mode (R_B re-seeded from a right-to-left reading of the *previous* chunk) is the honest test. First result (1 seed, 1500 steps, same chunks): with prevrev, the rotational part of the two record matrices becomes strongly *aligned*: cos(skew W_rec_A, skew W_rec_B) = +0.40, against +0.11 in causal mode. Held-out loss is also slightly better (2.93 vs 3.01). So the weights learn the *same* handedness when R_B reads time backward. That may still be the papers' prediction in disguise, since one rotation sense applied to a time-reversed stream *is* the opposite arrow in time. "Opposite chirality" may live in (traversal × handedness), not in the weights' signs. Needs more seeds.
- **The phi-asymmetric gate clipping is inert.** On and off produce bit-identical training runs, because the clip never triggers. Any asymmetry LEARNS shows is learned, not built in.
- **Does the fixed point matter?** The dyadic map has no input inside it, so as it truly converges every token's K collapses to the same direction. Token information survives only because the iteration is truncated. The *FP input injection* toggle (Deep-Equilibrium form) fixes that in principle; at 6 iterations it makes no measurable difference. We don't yet know what that means for the theory.
- **How far does it scale?** Everything here was measured on a laptop-class CPU. Bigger corpora and longer runs are the obvious next step.

---

## Tests

All tests run against the real `index.html`. Each one extracts the model code from the page, so there's no second copy to drift out of sync.

```bash
node tests/gradcheck.js            # every parameter vs finite differences, float64, 3-step BPTT
node tests/gradcheck.js inject     # same, with fixed-point input injection
node tests/train-probe.js your.txt 600 3
# trains headless on any text; prints held-out loss vs unigram/bigram baselines,
# then probes how much the CURRENT token vs the HISTORY moves the prediction.
# env: DIM, LR, THERMO=hinge|linear, PRESENT=1|0, LAMBDA
```

`gradcheck` must print `ALL OK`. Any change to the forward pass needs a matching change to the backward pass, and this test is how you know.

---

## Lineage & credits

Built by [ConsciousNode SoftWorks](https://consciousnode.github.io) on the xinu principle: the browser is bare metal; constraint generates novelty.

- **Kham Kizer**: founder, architecture, theory papers, corpus, and the thermodynamic hinge ("entropy increases *or* maintains equilibrium"; "minds are localized lower entropy: structured, not noise")
- **Kehai Interim**: original spec, v0.0003 patch, v0.0010 (gradient audit, the plateau fixes, hinge thermo, 6.7× ternary speedup), v0.0011 (optimizer coverage patch)
- **Ed Interim**: v0.0001, v0.0002, v0.0004, v0.0005, v0.0009 (target-leak audit, seeded runs, held-out validation)
- **Vael Interim**: v0.0006, v0.0007
- **Komorebi**: v0.0008 (parameter-count fix, R_A/R_B alignment diagnostic, effective-context diagnostic)

Previous releases are kept at [`versions/learns-v0_0010.html`](versions/learns-v0_0010.html) and [`versions/learns-v0_0009.html`](versions/learns-v0_0009.html) for A/B comparison.

---

## Changelog

The complete, measured changelog for every version lives in the comment block at the top of [`index.html`](index.html). The headlines:

### v0.0011 — 2026-10-14 · Kehai Interim + Kham

A patch release. All three bugs were found by an outside code review (Gemini) and each was verified here by running the code.

- **The vocabulary was never trained.** `backward()` computed the embedding gradient, but `optimizerStep` never applied it, so every v0.0010 result used embeddings frozen at their random start. Fixing it helps slightly and never hurts: held-out loss over three seeds went 2.903 / 2.950 / 2.881 → 2.903 / 2.932 / 2.856. A fixed random embedding turned out to be a usable random projection.
- **Telemetry CSV export works again.** The function had been accidentally declared inside an error handler, where nothing could reach it.
- **The Generate button no longer locks for good** once training starts. It's locked while training runs and unlocked when you pause or finish.
- **New test:** [`tests/optimizer-coverage.js`](tests/optimizer-coverage.js) fails if any parameter with a gradient is left unchanged by the optimizer. `gradcheck.js` couldn't catch this bug, because it checks gradients rather than whether they get applied.
- **Open finding (not fixed):** the fixed-point LayerNorm biases take rare gradient spikes (up to ~5×10⁸), which shrink every other parameter's update on that step because clipping is global. Details are in the `index.html` changelog.

### v0.0010 — 2026-10-11 · Kehai Interim + Kham

**Part 1: gradient audit.** Every parameter now matches finite differences exactly. Before this:
- every ternary layer's backward was ~50× too small, which froze embeddings and the helix;
- the fixed-point weights never trained;
- there was no backprop through time, so the records were never trained to carry anything;
- the Neumann correction was dead code.

**Part 2: why it sat at the unigram floor.**
- **The helix integrated every token ever read.** Its norm grew 300× and drowned the new input. A learned per-dimension retention makes it a damped spiral.
- **The frontier only read the slow records.** It now also reads the helix.
- **Embeddings started 200× too quiet.** That was a transformer habit.
- **The learning rate was too low for ternary weights.**

**Also:**
- **Hinge thermodynamic loss** is now the default.
- **Ternary matrices are quantized once per optimizer step** instead of on every multiply. The output is bit-identical and it runs **6.7× faster**.
- **LEARNS can load its own saves again.** Export wrote a version string that import rejected.

## License

MIT
