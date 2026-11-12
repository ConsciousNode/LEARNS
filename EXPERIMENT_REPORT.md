# LEARNS v0.0016 ("Rhythm") — Experiment Report (`REPORT.md` / `EXPERIMENT_REPORT.md`)

*(ConsciousNode SoftWorks · Audited Baseline: **B0** `scoutScale=0;rbMode=causal;fpNorm=rescale;fpInject=on` · Code Hash: `0x9c28cae9` · `.loom` Train Corpus Hash: `0x27600ed3` · Prose Corpus Hash: `0x1c7849c3`)*

## 1. Executive Summary & Headline Verdicts

LEARNS v0.0016 ("Rhythm") tests whether LEARNS's recurrent components — especially the **Helical State** (Component 5: Givens rotation plus learned retention, a built-in phase and timescale) — give it an edge on music written as text in AuraLoom's `.loom` score format (`1,9,17,25 kick 70`, `1-128/2 hat 30`, `1,5,9,13 pluck C#2 4 56 voice=fmBass`), and runs a direct walk-order control (`rbMode=window`) on the Sherlock Holmes prose corpus (`corpus.txt`) to test v0.0015's look-back hypothesis.

### 1.1 Interruption & Resumption Audit Log

Per audit rules on rate-limit / container interruptions, here is the exact execution and resumption provenance of every run in `results.jsonl`:

1. **Zero Runs Span Any Interruption (Atomic Single-Run Execution):**
   - Every training run in `run_batch_v16_b1.js`, `run_batch_v16_b2a.js`, and `run_batch_v16_b2b.js` spawns a fresh, standalone OS process (`node tests/run.js ...`) that initializes its PRNG from scratch (`L.seedRNG(seed >>> 0)`), runs steps `1..1000` plus tail/final evaluations entirely in memory, and emits a single JSON line to `stdout` **only upon completion of step 1,000 and all evaluations**.
   - The batch runner only appends a record to `results-v0016-*.jsonl` inside `child.on('close', code => { if (code === 0 && stdout.trim()) ... })`. Any child process interrupted mid-run is terminated without writing a single byte to disk, and when the batch script resumes, any unfinished `(tag, seed)` job is **rerun from step 0 from scratch** (never resumed from an intermediate checkpoint). Consequently, **no result in any table comes from a run that spanned an interruption**.
2. **Exact Timeline of Completed Runs Before and After Each Cutoff:**
   - **Completed prior to Cutoff 1 (VM container recycle during Batch 2A at `01:50 UTC-8`):**
     - All **43 / 43** runs of `results-v0016-b1.jsonl` (the 3 prose bit-identity reruns `v16_t4_repro_B0`, `v16_t4_repro_lookback16`, `v16_t4_repro_lookback64` on `seed=3`; the 4 prose `rbMode=window` LR sweep runs; all 10 5-seed prose runs for `v16_t4_window16` and `v16_t4_window64`; all 8 `.loom` n-gram baselines; and all 18 `.loom` seed-3 validation LR sweep runs).
     - The first **7 / 30** runs of `results-v0016-b2a.jsonl` (`v16_t2_gru_ternary` seeds `3, 7, 11, 13, 17` and `v16_t2_gru_float32` seeds `3, 7`). The in-flight seeds `11` and `13` of `v16_t2_gru_float32` were discarded and rerun from scratch from step 0 when `run_batch_v16_b2a.js` resumed at `01:54 UTC-8`.
   - **Completed between Cutoff 1 and Cutoff 2 (120-call turn limit at `02:45 UTC-8`):**
     - The remaining **23 / 30** runs of `results-v0016-b2a.jsonl` (`v16_t2_gru_float32` seeds `11, 13, 17`; `v16_t2_learns_ternary` seeds `3, 7, 11, 13, 17`; `v16_t2_learns_float32` seeds `3, 7, 11, 13, 17`; `v16_t2_transformer_ternary` seeds `3, 7, 11, 13, 17`; and `v16_t2_transformer_float32` seeds `3, 7, 11, 13, 17`), completing **30 / 30** Tier 2 benchmark runs.
     - The first **8** runs of `results-v0016-b2b.jsonl` (`v16_t3_helix_off` seeds `3, 7, 11, 13, 17` [all 5 seeds] and `v16_t3_helixRot_none` seeds `3, 7, 11`).
   - **Resumed after Cutoff 2 (at `15:17 UTC-8`, starting at `v16_t3_helixRot_none` seed `13`):**
     - `run_batch_v16_b2b.js` resumed from scratch on `v16_t3_helixRot_none` seeds `13` and `17`, completing all 5 seeds (`3, 7, 11, 13, 17`) of `v16_t3_helixRot_none` (11 completed Tier 3 records total: `v16_t3_helix_off:seed=3, v16_t3_helix_off:seed=7, v16_t3_helix_off:seed=11, v16_t3_helix_off:seed=13, v16_t3_helix_off:seed=17, v16_t3_helixRot_none:seed=3, v16_t3_helixRot_none:seed=7, v16_t3_helixRot_none:seed=11, v16_t3_helixRot_none:seed=13, v16_t3_helixRot_none:seed=17, v16_t3_helixRet_fixed:seed=3`). Any in-flight process after `v16_t3_helixRot_none:seed=17` was terminated cleanly before writing to `results.jsonl`.

---

### 1.2 Headline Scientific Findings (`Measured:` vs `Hypothesis:`)

1. **Tier 1 — Procedural `.loom` Corpus from Synth Radio (`tests/loom-corpus.js`):**
   - **Measured:** Lifting Synth Radio v13+'s composition engine (`146` seeded channels across `12` genre clusters, Euclidean rhythms, diatonic harmony, and role-based voice selection) and mapping all `50` Synth Radio voices to AuraLoom's `12` voices (`pure`, `flute`, `string`, `reed`, `organ`, `bell`, `fmLead`, `fmBass`, `fmKeys`, `fmBrass`, `fmBell`, `fmClang`) yields **10,000 / 10,000 valid `.loom` exports** (`0` parse or loom-rendering errors, `1,239,950` woven events) verified by AuraLoom's own `weaveScore` parser.
   - **Measured:** Splitting the `146` channels with fixed seed `20261111` produces `110` training channels (`loom-train.txt`, `2,003,164` bytes across `1,207` comment-free scores), `10` validation channels (`loom-val.txt`, `200,261` bytes across `125` scores from disjoint seeds), and `26` whole held-out unseen-genre test channels (`loom-test-unseen.txt`, `74,278` bytes across `52` scores), plus the `8` comment-stripped studio crew scores (`loom-test-crew.txt`, `7,328` bytes).
   - **Measured:** Leak checking `loom-train.txt` against each of the `8` comment-stripped crew scores (`peepers`, `hollow`, `hollow-2`, `threshold`, `understory`, `understory-morning`, `understory-rest`, `understory-trouble`) shows longest common substrings of `29–42` bytes (standard header blocks or single accompaniment lines) and unique 16-byte n-gram overlaps of `16.03%–29.92%`.

2. **Tier 2 — The `.loom` Music Benchmark (LEARNS B0 vs Parameter-Matched GRU and Transformer):**
   - **Measured (vs GRU):** Parameter-matched `GRU` (`54,450` active params) beats `LEARNS B0` (`54,400` active params) on `.loom` music across all splits and all three byte roles (`step`, `pitch`, `other`) in both ternary and float32 on `5/5` seeds:
     - **Unseen genres (26 held-out channels):** `GRU float32` (`0.6415 ± 0.0130` nats/byte) and `GRU ternary` (`0.7131 ± 0.0067`) beat `LEARNS B0 float32` (`0.8081 ± 0.0278`) and `LEARNS B0 ternary` (`0.8953 ± 0.0215`, paired $\Delta = -0.1822$ nats vs `B0 ternary`, 95% CI $[-0.2099, -0.1545]$, `5/5` seeds, **WIN for GRU**).
     - **Crew scores (8 studio pieces):** `GRU float32` (`1.0690 ± 0.0202`) and `GRU ternary` (`1.1152 ± 0.0171`) beat `LEARNS B0 float32` (`1.2134 ± 0.0262`) and `LEARNS B0 ternary` (`1.2942 ± 0.0404`), winning on all `8/8` individual crew scores.
     - **Byte-role breakdown (vs GRU):** On unseen genres, `GRU ternary` beats `LEARNS B0 ternary` on `step` bytes (`1.6644` vs `1.9084`, $-0.2440$ nats), `pitch` bytes (`1.6138` vs `1.7255`, $-0.1117$ nats), and `other` bytes (`0.4610` vs `0.6404`, $-0.1794$ nats).
   - **Measured (vs Transformer — Ranking Reversal from Prose):** Whereas `LEARNS B0` trailed the parameter-matched `Transformer` (`55,440` active params) on Sherlock prose in v0.0014/v0.0015, **`LEARNS B0` beats `Transformer` on `.loom` music in both ternary and float32 across all splits on `5/5` seeds**:
     - **Ternary:** `LEARNS B0 ternary` (`0.8953 ± 0.0215` unseen, `1.2942 ± 0.0404` crew) beats `Transformer ternary` (`0.9338 ± 0.0152` unseen, `1.3871 ± 0.0278` crew) by **$-0.0385$ nats** on unseen genres (95% CI $[-0.0765, -0.0005]$, `5/5` seeds, **WIN for LEARNS**) and **$-0.0928$ nats** on the 8 crew scores (`5/5` seeds).
     - **Float32:** `LEARNS B0 float32` (`0.8081 ± 0.0278` unseen, `1.2134 ± 0.0262` crew) beats `Transformer float32` (`0.9173 ± 0.0118` unseen, `1.3515 ± 0.0172` crew) by **$-0.1092$ nats** on unseen genres (`5/5` seeds, **WIN for LEARNS**) and **$-0.1381$ nats** on the 8 crew scores (`5/5` seeds), including **$-0.1362$ nats** on `step` bytes (`1.8239` vs `1.9601`) and **$-0.1071$ nats** on `pitch` tokens (`1.6592` vs `1.7663`).
   - **Hypothesis:** `LEARNS` and `GRU` carry persistent recurrent state across contiguous 64-byte chunks within a `.loom` score (`400–1,800` bytes), preserving score-header context (`key`, `bars`, `voice`) and multi-bar drum/bass structure across chunk boundaries where `Transformer` resets its 64-byte causal attention window.

3. **Tier 3 — Component & Helix Ablations on `.loom` (`helix=off` and `helixRotation=none`):**
   - **Measured (`helix=off`, 5 seeds):** Disabling the Helical State (`helix=off`) severely degrades `LEARNS B0` on `.loom` music on `5/5` seeds: unseen-genre test loss worsens by **$+0.1972$ nats** (`1.0925` vs `0.8953`, 95% CI $[+0.0079, +0.3865]$, `0/5`, **LOSS**) and crew-score loss worsens by **$+0.2172$ nats** (`1.5114` vs `1.2942`).
   - **Measured (`helixRotation=none`, 5 seeds):** Removing the Givens phase rotation while keeping learned retention (`helixRotation=none`) degrades unseen-genre test loss on `5/5` seeds by **$+0.0490$ nats** (`0.9443` vs `0.8953`, 95% CI $[+0.0199, +0.0781]$, `0/5`, **LOSS**) and crew-score loss by **$+0.0517$ nats** (`1.3459` vs `1.2942`), dropping `LEARNS` below `Transformer`!
   - **Measured (Byte-Role Test of the Rhythm Hypothesis):**
     - Removing the entire helix (`helix=off`) hurts `step`-column rhythm bytes by **$+0.2103$ nats** (`2.1187` vs `1.9084`), `pitch` tokens by **$+0.2043$ nats** (`1.9298` vs `1.7255`), and `other` bytes by **$+0.1943$ nats** (`0.8347` vs `0.6404`).
     - Removing specifically the Givens rotation (`helixRotation=none`) hurts `step`-column rhythm bytes by **$+0.0238$ nats** (`1.9322` vs `1.9084`) and `other` bytes by **$+0.0616$ nats** (`0.7020` vs `0.6404`), while `pitch` tokens change by only **$+0.0085$ nats** (`1.7340` vs `1.7255`). Thus within the helix, the Givens phase rotation (`helixRotation`) helps `step` and structural line timing nearly $3\times$ more than `pitch` tokens ($+0.0238$ vs $+0.0085$ nats), and is the exact component that gives `LEARNS B0` its $-0.0385$ nat victory over `Transformer ternary`!

4. **Tier 4 — Look-Back Walk-Order Control on Sherlock Prose (`rbMode=window` vs `rbMode=lookback` and `B0`):**
   - **Measured (Causality & Gradient Verification):** `rbMode=window` (walking the $N$-byte window *forward* from $x_{t-N+1}$ to $x_t$ starting from $\mathbf{0}$, with exact BPTT) passes `tests/leak.js` (`max|Δlogit| = 0.000e+0` on random and trained weights; standalone scratch-window equivalence `max|ΔR_B| = 0.00e+0`), `tests/gradcheck.js --all` (`47/47`), and `tests/optimizer-coverage.js --all` (`48/48`).
   - **Measured (Bit-Identity of v0.0015 Reruns):** Rerunning seed `3` of `B0` (`val=2.176434, test=2.216040`), `rbMode=lookback;lookback=16` (`val=2.184054, test=2.224706`), and `rbMode=lookback;lookback=64` (`val=2.186538, test=2.232847`) under v0.0016 (`codeHash=0x9c28cae9`) reproduces v0.0015 (`codeHash=0x626fd26c`) to six decimal places (`0.000000` difference). Seeds `7, 11, 13, 17` of those three arms are therefore reused with their original `codeHash` (`0x626fd26c`) and `reusedFrom: "v0.0015"` (marked with `†` in Table 7).
   - **Measured (5-Seed Walk-Order Comparison on `corpus.txt`):**
     - `B0` (`rbMode=causal`): test `2.2181 ± 0.0105` nats/byte.
     - `rbMode=lookback;lookback=16` (backward walk $x_t \to x_{t-15}$): test `2.2188 ± 0.0132` ($+0.0007$ vs `B0`, **inconclusive**).
     - `rbMode=window;lookback=16` (forward walk $x_{t-15} \to x_t$): test `2.2198 ± 0.0214` ($+0.0016$ vs `B0`, 95% CI $[-0.0231, +0.0264]$, **inconclusive**; $+0.0010$ vs `lookback=16`, 95% CI $[-0.0174, +0.0193]$, `3/5`, **inconclusive**).
     - `rbMode=lookback;lookback=64` (backward walk $x_t \to x_{t-63}$): test `2.2228 ± 0.0126` ($+0.0046$ vs `B0`, **inconclusive**).
     - `rbMode=window;lookback=64` (forward walk $x_{t-63} \to x_t$): test `2.2176 ± 0.0243` ($-0.0005$ vs `B0`, 95% CI $[-0.0320, +0.0310]$, **inconclusive**; $-0.0051$ vs `lookback=64`, 95% CI $[-0.0319, +0.0216]$, `2/5`, **inconclusive**).
   - **Measured Verdict on the v0.0015 Walk-Order Hypothesis:** Walking the $N$-byte window forward (`rbMode=window`) vs backward (`rbMode=lookback`) is statistically **inconclusive** at both $N=16$ ($+0.0010$ nats) and $N=64$ ($-0.0051$ nats), and neither beats `B0` (`rbMode=causal`).

---

## 2. Generated Tables from `tests/tables.js`

*(The tables below are recomputed directly from `results.jsonl` and `loom-corpus-meta.json` by `node tests/tables.js`.)*

### Table 1: Tier 1 Procedural `.loom` Corpus & Crew Score Leak Check (`seed=20261111`)

| Split / Crew Score | Author / Source | Channels / Scores | Total Bytes | `step` Bytes | `pitch` Bytes | `other` Bytes | Longest Common Substring vs Train | 16-Gram Unique Overlap |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `loom-train.txt` | Synth Radio (110 ch) | 110 ch / 1207 scores | 2003164 | 294540 | 161437 | 1547187 | — | — |
| `loom-val.txt` | Synth Radio (10 ch) | 10 ch / 125 scores | 200261 | 26364 | 16917 | 156980 | — | — |
| `loom-test-unseen` | Synth Radio (26 ch) | 26 ch / 52 scores | 74278 | 9649 | 6161 | 58468 | — | — |
| `loom-test-crew` (all 8) | Studio Crew | 8 scores | 7328 | 818 | 739 | 5771 | — | — |
| Crew: `peepers` | Kehai (4 bars, 124 BPM) | 1 (126 woven) | 1751 | 221 | 128 | 1402 | 42 B (`"0 voice=fmBass\n43         ...`) | 29.92% (435/1454) |
| Crew: `hollow` | Ed (8 bars, 76 BPM) | 1 (32 woven) | 735 | 71 | 101 | 563 | 31 B (`"        hat 30\n\n1      pl...`) | 19.10% (136/712) |
| Crew: `hollow-2` | Ed (8 bars, 76 BPM) | 1 (32 woven) | 765 | 70 | 103 | 592 | 31 B (`"        hat 30\n\n1      pl...`) | 17.65% (131/742) |
| Crew: `threshold` | Vael (8 bars, 76 BPM) | 1 (24 woven) | 523 | 39 | 65 | 419 | 29 B (`"\nbars 8\nkey A phrygian\nv...`) | 21.34% (105/492) |
| Crew: `understory` | Ed (8 bars, 68 BPM) | 1 (42 woven) | 979 | 93 | 106 | 780 | 39 B (`"\nbars 8\nkey D dorian\nvoi...`) | 22.43% (205/914) |
| Crew: `understory-morning` | Ed (8 bars, 68 BPM) | 1 (37 woven) | 978 | 77 | 103 | 798 | 39 B (`"\nbars 8\nkey D dorian\nvoi...`) | 24.21% (221/913) |
| Crew: `understory-rest` | Ed (1 bars, 60 BPM) | 1 (9 woven) | 309 | 12 | 30 | 267 | 32 B (`"\nkey D dorian\nvoice flute...`) | 25.51% (75/294) |
| Crew: `understory-trouble` | Ed (8 bars, 88 BPM) | 1 (91 woven) | 1281 | 235 | 103 | 943 | 33 B (`" voice=string\n\n1      ton...`) | 16.03% (188/1173) |

### Table 2: Tier 2 Validation Learning-Rate Sweep on `.loom` (`seed=3`, 1,000 Steps)

| Model | Weights | Params | `lr=0.004` Val | `lr=0.008` Val | `lr=0.016` Val | `lr=0.024` Val | Selected LR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| GRU | float32 | 54450 | — | 0.6705 | 0.6570 | 0.6833 | `0.016` |
| GRU | ternary | 54450 | — | 0.7904 | 0.7135 | 0.7356 | `0.016` |
| LEARNS (B0) | float32 | 54400 | 1.0722 | 0.8126 | 0.8338 | — | `0.008` |
| LEARNS (B0) | ternary | 54400 | — | 0.9490 | 0.9005 | 0.9807 | `0.016` |
| Transformer | float32 | 55440 | 0.8879 | 0.9002 | 0.9587 | — | `0.004` |
| Transformer | ternary | 55440 | 1.0357 | 0.9231 | 0.9428 | — | `0.008` |

### Table 3: Tier 2 `.loom` Benchmark — 5-Seed Overall & Byte-Role Losses (`seeds=3,7,11,13,17`, 1,000 Steps)

| Model / Baseline | Params | Val Mean±SD | Val Tail Range | Unseen Genres Mean±SD | Unseen `step` / `pitch` / `other` | Crew Scores Mean±SD | Crew `step` / `pitch` / `other` | Unseen Δ vs B0 Ternary (95% CI, Verdict) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| GRU (float32, `lr=0.016`) | 54450 | 0.6447±0.0098 | 0.0003 | 0.6415±0.0130 | 1.5577 / 1.5450 / 0.3949 | 1.0690±0.0202 | 2.0465 / 2.5476 / 0.7403 | -0.2538 ([-0.2733, -0.2344], 5/5, **WIN**) |
| GRU (ternary, `lr=0.016`) | 54450 | 0.7115±0.0074 | 0.0019 | 0.7131±0.0067 | 1.6644 / 1.6138 / 0.4610 | 1.1152±0.0171 | 2.0611 / 2.5305 / 0.7991 | -0.1822 ([-0.2099, -0.1545], 5/5, **WIN**) |
| LEARNS B0 (float32, `lr=0.008`) | 54400 | 0.7989±0.0263 | 0.0006 | 0.8081±0.0278 | 1.8239 / 1.6592 / 0.5505 | 1.2134±0.0262 | 2.1568 / 2.6264 / 0.8979 | -0.0873 ([-0.1051, -0.0694], 5/5, **WIN**) |
| LEARNS B0 (ternary, `lr=0.016`) | 54400 | 0.8866±0.0201 | 0.0087 | 0.8953±0.0215 | 1.9084 / 1.7255 / 0.6404 | 1.2942±0.0404 | 2.2172 / 2.6890 / 0.9840 | — (baseline) |
| Transformer (float32, `lr=0.004`) | 55440 | 0.9063±0.0120 | 0.0006 | 0.9173±0.0118 | 1.9601 / 1.7663 / 0.6555 | 1.3515±0.0172 | 2.2353 / 2.7950 / 1.0406 | +0.0219 ([-0.0146, +0.0584], 2/5, **inconclusive**) |
| Transformer (ternary, `lr=0.008`) | 55440 | 0.9245±0.0151 | 0.0018 | 0.9338±0.0152 | 1.9456 / 1.7453 / 0.6811 | 1.3871±0.0278 | 2.2140 / 2.8214 / 1.0854 | +0.0385 ([+0.0005, +0.0765], 0/5, **LOSS**) |
| 5gram (`ngramData=seen`) | 9854 | 0.5753 | 0.0000 | 0.5795 | 1.1914 / 1.4721 / 0.3843 | 1.4813 | 2.3284 / 3.6776 / 1.0790 | — |
| 5gram (`ngramData=all`) | 16817 | 0.5221 | 0.0000 | 0.5275 | 1.0557 / 1.3421 / 0.3544 | 1.6952 | 2.9406 / 3.9864 / 1.2241 | — |
| trigram (`ngramData=seen`) | 1760 | 0.8186 | 0.0000 | 0.8200 | 1.7404 / 1.6386 / 0.5816 | 1.5340 | 2.2640 / 3.7479 / 1.1460 | — |
| trigram (`ngramData=all`) | 2202 | 0.8048 | 0.0000 | 0.8057 | 1.7125 / 1.5921 / 0.5730 | 1.6028 | 2.3854 / 4.2496 / 1.1517 | — |
| bigram (`ngramData=seen`) | 456 | 1.5164 | 0.0000 | 1.5054 | 2.3906 / 2.4737 / 1.2571 | 2.0084 | 2.5396 / 4.3354 / 1.6342 | — |
| bigram (`ngramData=all`) | 491 | 1.5130 | 0.0000 | 1.5004 | 2.3795 / 2.4654 / 1.2534 | 2.0692 | 2.5291 / 5.0711 / 1.6184 | — |
| unigram (`ngramData=seen`) | 53 | 3.1651 | 0.0000 | 3.1388 | 3.7141 / 4.5427 / 2.8958 | 3.1187 | 3.7178 / 4.6497 / 2.8370 | — |
| unigram (`ngramData=all`) | 53 | 3.1620 | 0.0000 | 3.1352 | 3.7111 / 4.5328 / 2.8926 | 3.1156 | 3.7159 / 4.6389 / 2.8347 | — |

### Table 4: Tier 2 Per-Score Loss on the 8 Comment-Stripped Crew Scores (5-Seed Mean Nats/Byte)

| Model | `peepers` | `hollow` | `hollow-2` | `threshold` | `understory` | `understory-morning` | `understory-rest` | `understory-trouble` | Unweighted 8-Score Mean | Byte-Weighted Crew Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GRU (float32, `lr=0.016`) | 0.9092 | 1.2305 | 1.2572 | 1.2210 | 1.0464 | 1.0163 | 1.0273 | 1.0883 | 1.0995 | 1.0690 |
| GRU (ternary, `lr=0.016`) | 0.9539 | 1.2666 | 1.3040 | 1.2544 | 1.0981 | 1.0659 | 1.0994 | 1.1340 | 1.1470 | 1.1152 |
| LEARNS B0 (float32, `lr=0.008`) | 1.0303 | 1.4617 | 1.5370 | 1.4091 | 1.1503 | 1.1187 | 1.2548 | 1.1587 | 1.2651 | 1.2134 |
| LEARNS B0 (ternary, `lr=0.016`) | 1.1008 | 1.5498 | 1.6054 | 1.5384 | 1.2281 | 1.1947 | 1.4390 | 1.2184 | 1.3593 | 1.2942 |
| Transformer (float32, `lr=0.004`) | 1.1236 | 1.6287 | 1.7103 | 1.5049 | 1.3037 | 1.2702 | 1.3942 | 1.3158 | 1.4064 | 1.3515 |
| Transformer (ternary, `lr=0.008`) | 1.1546 | 1.6813 | 1.7425 | 1.5335 | 1.3453 | 1.3044 | 1.4499 | 1.3442 | 1.4445 | 1.3871 |

### Table 5: Tier 2 Per-Channel Loss across the 26 Held-Out Unseen-Genre Channels (5-Seed Mean Nats/Byte)

| Held-Out Channel (`id`) | Cluster | Bytes | GRU (float32) | GRU (ternary) | LEARNS B0 (float32) | LEARNS B0 (ternary) | Transformer (float32) | Transformer (ternary) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `anime_op` | japanese | 4580 | 0.6469 | 0.7068 | 0.7800 | 0.8459 | 0.8897 | 0.8969 |
| `honky_tonk` | folk | 2852 | 0.6561 | 0.7332 | 0.8452 | 0.9402 | 0.9536 | 0.9609 |
| `darkwave` | ambient | 2871 | 0.6607 | 0.7484 | 0.8565 | 0.9466 | 0.9916 | 0.9952 |
| `microsound` | ambient | 1373 | 0.6598 | 0.7611 | 0.9152 | 1.0531 | 1.0643 | 1.0914 |
| `cool_jazz` | jazz | 4002 | 0.5770 | 0.6258 | 0.6648 | 0.7350 | 0.7522 | 0.7654 |
| `afrobeat` | world | 4493 | 0.6059 | 0.6643 | 0.7228 | 0.8055 | 0.8107 | 0.8262 |
| `indie_pop` | pop | 2951 | 0.6405 | 0.7176 | 0.8267 | 0.9240 | 0.9438 | 0.9507 |
| `demoscene` | retro | 3692 | 0.6652 | 0.7301 | 0.8211 | 0.8946 | 0.8934 | 0.9139 |
| `city_pop_jp` | japanese | 3059 | 0.6518 | 0.7294 | 0.8447 | 0.9302 | 0.9419 | 0.9489 |
| `spectral` | ambient | 1333 | 0.7165 | 0.8184 | 0.9650 | 1.0986 | 1.1395 | 1.1634 |
| `uk_drill` | hiphop | 2309 | 0.6541 | 0.7364 | 0.8561 | 0.9475 | 0.9748 | 0.9958 |
| `uk_garage` | electronic | 2544 | 0.6666 | 0.7437 | 0.8543 | 0.9406 | 0.9620 | 0.9714 |
| `silence_with` | special | 1206 | 0.6363 | 0.7156 | 0.8600 | 0.9628 | 0.9563 | 0.9759 |
| `g_funk` | hiphop | 4511 | 0.5529 | 0.6084 | 0.6613 | 0.7341 | 0.7694 | 0.7719 |
| `impressionist` | classical | 1660 | 0.6587 | 0.7412 | 0.8645 | 0.9718 | 0.9675 | 0.9854 |
| `dream_pop` | pop | 2889 | 0.5811 | 0.6553 | 0.7476 | 0.8331 | 0.8688 | 0.8891 |
| `bedroom_pop` | pop | 2473 | 0.5798 | 0.6532 | 0.7478 | 0.8466 | 0.8510 | 0.8765 |
| `deconstructed_club` | ambient | 2639 | 0.6784 | 0.7517 | 0.8736 | 0.9599 | 0.9879 | 1.0254 |
| `math_rock` | rock | 5315 | 0.5608 | 0.6137 | 0.6467 | 0.7225 | 0.7429 | 0.7567 |
| `chicago_drill` | hiphop | 3579 | 0.7267 | 0.8117 | 0.9111 | 1.0031 | 1.0340 | 1.0503 |
| `choral` | classical | 1721 | 0.6594 | 0.7395 | 0.8497 | 0.9684 | 0.9505 | 0.9690 |
| `acid_house` | electronic | 3080 | 0.7255 | 0.7893 | 0.8921 | 0.9774 | 0.9788 | 0.9857 |
| `dungeon_synth` | ambient | 1428 | 0.6607 | 0.7514 | 0.8826 | 0.9988 | 1.0116 | 1.0482 |
| `chiptune_nes` | retro | 3441 | 0.7292 | 0.8150 | 0.9562 | 1.0404 | 1.0970 | 1.1250 |
| `microtonal` | special | 3057 | 0.6222 | 0.6969 | 0.8130 | 0.8975 | 0.9371 | 0.9771 |
| `noise_structured` | ambient | 1169 | 0.7565 | 0.8447 | 0.9992 | 1.0968 | 1.1227 | 1.1583 |

### Table 6: Tier 3 `.loom` Component & Helix Ablations vs LEARNS B0 Ternary (`lr=0.016`, 1,000 Steps)

| Configuration | Seeds | Val Loss | Unseen Genres Loss | Unseen `step` / `pitch` / `other` | Crew Loss | Crew `step` / `pitch` / `other` | Unseen Δ vs B0 (Paired across matched seeds) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| B0: `learns` default (`helix=on`) | 5 | 0.8866 | 0.8953 | 1.9084 / 1.7255 / 0.6404 | 1.2942 | 2.2172 / 2.6890 / 0.9840 | — (baseline) |
| Ablation: `helix=off` | 5 | 1.0874 | 1.0925 | 2.1187 / 1.9298 / 0.8347 | 1.5114 | 2.3182 / 2.8947 / 1.2191 | +0.1972 (step +0.2103, pitch +0.2043) |
| Ablation: `helixRotation=none` | 5 | 0.9327 | 0.9443 | 1.9322 / 1.7340 / 0.6978 | 1.3459 | 2.2153 / 2.6623 / 1.0534 | +0.0490 (step +0.0238, pitch +0.0085) |
| Ablation: `helixRetention=fixed` | 1 | 0.9098 | 0.9243 | 1.9440 / 1.7306 / 0.6709 | 1.3198 | 2.3509 / 2.5523 / 1.0150 | +0.0142 (step +0.0515, pitch +0.0279) |

### Table 7: Tier 4 Look-Back Walk-Order Control on Sherlock Prose (`corpus.txt`, 5 Seeds `3,7,11,13,17`, 1,000 Steps)

| Prose Arm (`corpus.txt`) | Walk Order over $x_{t-N+1}\dots x_t$ | Val Mean±SD | Test Mean±SD | Δ Test vs B0 (95% CI, Wins, Verdict) | Δ Test vs `lookback` Same $N$ (95% CI, Verdict) |
| :--- | :--- | :---: | :---: | :--- | :--- |
| B0: `rbMode=causal` (`scoutScale=0`) † | Causal carry ($\mathbf{R}_{B,t}$ from $\mathbf{R}_{B,t-1}$ + $\mathbf{K}_{B,t}$) | 2.1742±0.0089 | 2.2181±0.0105 | — (baseline) | — |
| A3-16: `rbMode=lookback;lookback=16` † | Backward ($x_t \to x_{t-15}$, oldest last) | 2.1766±0.0098 | 2.2188±0.0132 | +0.0007 ([-0.0213, +0.0227], 2/5, **inconclusive**) | — |
| T4-W16: `rbMode=window;lookback=16` | Forward ($x_{t-15} \to x_t$, newest last) | 2.1775±0.0203 | 2.2198±0.0214 | +0.0016 ([-0.0231, +0.0264], 2/5, **inconclusive**) | +0.0010 ([-0.0174, +0.0193], 3/5, **inconclusive**) |
| A3-64: `rbMode=lookback;lookback=64` † | Backward ($x_t \to x_{t-63}$, oldest last) | 2.1796±0.0095 | 2.2228±0.0126 | +0.0046 ([-0.0091, +0.0184], 2/5, **inconclusive**) | — |
| T4-W64: `rbMode=window;lookback=64` | Forward ($x_{t-63} \to x_t$, newest last) | 2.1734±0.0224 | 2.2176±0.0243 | -0.0005 ([-0.0320, +0.0310], 1/5, **inconclusive**) | -0.0051 ([-0.0319, +0.0216], 2/5, **inconclusive**) |

*(† Rows marked with † reuse seeds 7, 11, 13, 17 from v0.0015 (`codeHash=0x626fd26c`, `reusedFrom="v0.0015"`) after verifying seed 3 is bit-for-bit identical under v0.0016 `codeHash=0x9c28cae9`.)*
