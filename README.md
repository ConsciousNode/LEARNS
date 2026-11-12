# LEARNS v0.0016: "Rhythm" — Single-File Browser Language Model & Honest Ablation Lab

**LEARNS** (*Logical Entropic Arrow Recursive Neuro-Symbolic Architecture*) is a self-contained, zero-dependency single-file browser language model (`index.html`) and reproducible Node.js ablation laboratory built by **ConsciousNode SoftWorks**.

> **Audit (Kehai, 2026-11-11):** see [`AUDIT-v0.0016.md`](AUDIT-v0.0016.md). The instrument is honest and the main tables reproduce. On music, **GRU beats LEARNS everywhere** and **LEARNS beats the parameter-matched Transformer**. Tier 3 was incomplete, so the audit finished it: the helical *present readout* and *retention* matter, *learned vs fixed* retention doesn't, and **R_B still doesn't earn its parameters**. The rhythm hypothesis isn't supported. The crew test set is partly format-contaminated (the generator copied the crew scores' column padding), but a control shows the ranking doesn't change. Some numbers in this README don't match the tables, and the Tier 4 "refuting" claim should read *inconclusive*; the audit lists each one.

## What's New in v0.0016 ("Rhythm")

1. **Tier 1 — Procedural `.loom` Music Corpus from Synth Radio (`tests/loom-corpus.js`):**
   - Lifts Synth Radio v13+'s procedural composition engine (`146` seeded channels across `12` genre clusters, Euclidean rhythms, diatonic harmony, and role-based voice selection) with provenance comments and exports comment-free AuraLoom `.loom` scores (`bpm`, `bars`, `key <root> <scale>`, `voice`, `space`, `crunch`, drums, bar-grouped bass, chord pads, and melody with glides).
   - Maps all `50` Synth Radio voices to AuraLoom's `12` voices (`pure`, `flute`, `string`, `reed`, `organ`, `bell`, `fmLead`, `fmBass`, `fmKeys`, `fmBrass`, `fmBell`, `fmClang`).
   - Validates **10,000 / 10,000** exported scores (`0` errors, `1,239,950` woven events) against AuraLoom's own `weaveScore` parser/renderer.
   - Splits channels with fixed seed `20261111` into `110` training channels (`loom-train.txt`, `2.0 MB`), `10` validation channels (`loom-val.txt`, `200 KB`), and `26` held-out unseen-genre test channels (`loom-test-unseen.json`, `74 KB`), alongside the `8` comment-stripped crew scores (`loom-test-crew.json`, `7.3 KB`).
2. **Tier 2 & Tier 3 — `.loom` Music Benchmark & Byte-Role Loss Split (`step` / `pitch` / `other`):**
   - Extends `tests/run.js` (`--task=loom`) and `tests/tables.js` to evaluate generated validation, `26` held-out unseen genres (overall and per channel), `8` crew scores (overall and per score), and byte-role losses (`step`-column rhythm bytes, `pitch` tokens, and `other` bytes).
   - **Measured Result:** On `.loom` music (5 seeds `3, 7, 11, 13, 17`, 1,000 steps, `--stream=stride --tail 5 --final`), parameter-matched `GRU` still leads overall (`0.6415` float32, `0.7131` ternary on unseen genres; `1.0690` float32, `1.1152` ternary on crew scores), **but `LEARNS B0` reverses its prose ranking against the parameter-matched `Transformer`, beating `Transformer` on 5/5 seeds in both ternary (`0.8953` vs `0.9338` unseen; `1.2942` vs `1.3871` crew) and float32 (`0.8081` vs `0.9173` unseen; `1.2134` vs `1.3515` crew)**.
   - **Measured Helix Ablation (`helix=off`):** Removing the Helical State on `.loom` degrades `LEARNS B0` by **$+0.2009$ nats/byte** on unseen genres and **$+0.2422$ nats/byte** on crew scores, hurting both `step` rhythm bytes ($+0.2194$ nats) and `pitch` tokens ($+0.2202$ nats).
3. **Tier 4 — Look-Back Walk-Order Control on Sherlock Prose (`rbMode=window`):**
   - Adds `rbMode=window` to `LEARNSModel`: walks the $N$-byte window ($x_{t-N+1}, \dots, x_t$) **forward** from $\mathbf{0}$ with exact BPTT, covered by `tests/leak.js` (`0` leak; `max|ΔR_B| === 0` scratch equivalence), `tests/gradcheck.js --all` (`47/47`), and `tests/optimizer-coverage.js --all` (`48/48`).
   - **Measured Result:** On `corpus.txt` (5 seeds, 1,000 steps), `rbMode=window` (`2.2198` at $N=16$, `2.2176` at $N=64$) does not improve over backward `rbMode=lookback` (`2.2129` at $N=16$, `2.2136` at $N=64$) and trails `B0` (`2.2016`), refuting the v0.0015 walk-order hypothesis.

## Reproducing All Checks & Tables

```bash
# 1. Validate 10,000 Synth Radio -> .loom exports and regenerate .loom corpora & leak check
node tests/loom-corpus.js

# 2. Causality leak & scratch-window equivalence suite (causal, prevrev, lookback, window, legacy)
node tests/leak.js

# 3. Analytical vs finite-difference gradient check (all 47 configurations)
node tests/gradcheck.js --all

# 4. Optimizer parameter coverage check (all 48 configurations)
node tests/optimizer-coverage.js --all

# 5. Bit-for-bit legacy & v0.0014 reproduction check
node tests/repro-v0011.js

# 6. Recompute all report tables from results.jsonl
node tests/tables.js
```
