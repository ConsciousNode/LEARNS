# LEARNS v0.0015 "The Partner Record": audit (Kehai, 2026-11-11)

## Verdict

**Clean.** Everything reproduces, every rule from the v0.0014 audit was followed, and I found no fabricated or relabelled result. This is the most trustworthy LEARNS release so far. Nothing needed fixing.

## What I checked

| Check | Result |
|---|---|
| `CODE_HASH` of the delivered `index.html` | `0x626fd26c`, matching every fresh record |
| `node tests/leak.js` | All causal modes 0 leak on random and trained weights; **`rbMode=legacy` fails as it must** (1.087e-1 / 3.864e-2, matching the report) |
| Does the positive control use the real code? | Yes. `trainChunk` itself swaps in the future-scout `R_B` only when `rbMode === 'legacy'`, so the test exercises the model's own leak path, not one built by the harness |
| Leak-freedom of `trainChunk` (read by hand, because `leak.js` exercises `forward()`) | `causal` and `lookback` use only the forward state; `lookback` walks only `tokenHistory` (bytes already seen); `prevrev` seeds from the previous chunk only when chunks are contiguous |
| `node tests/gradcheck.js --all` | 45/45 OK, including both lookback windows and `scoutHead=separate` |
| `node tests/optimizer-coverage.js --all` | 46/46 OK |
| `node tests/repro-v0011.js` | All OK. The pinned v0.0014 values match the audited v0.0014 record in the repo, not just pins written by Argon |
| `node tests/tables.js` parity | All 61 table rows in README and EXPERIMENT_REPORT regenerate verbatim from `results.jsonl` |
| Provenance (the v0.0014 problem) | All 233 v0.0014-file records are preserved unchanged. All 233 reused records carry `reusedFrom` and their original `codeHash` (218 × `0x4048aacb`, 15 × `0xdf8ae499`), and every one exists in the audited v0.0014 file. 43 records are fresh under `0x626fd26c` |
| Three fresh GRU 7,030-step records are numerically identical to v0.0014 ones | **Genuine reruns, not relabels.** Wall-clock and tokens/sec differ (for example 160 s vs 306 s), and GRU code is unchanged, so they reproduce bit for bit |
| Bit-for-bit reruns (`results-audit-v0015.jsonl`) | A3a `rbMode=lookback;lookback=16` seed 3, and B0 seed 7, both 1,000 steps with `--tail 5 --final`. **All 36 deterministic fields match** (test 2.224706 and 2.22552). The only difference is Argon's `tag` annotation |
| `tests/train-probe.js` | Its slicing markers all still exist and it runs |

## Findings worth knowing

1. **The A2 (`prevrev`) arm can't be tested under `--stream=stride`.** Stride chunks are never contiguous, so `prevrev` never engages in training. A2's training is bit-identical to B0's, and its eval differs by at most about 2e-5. The report says this openly. To test `prevrev` for real, rerun it under `--stream=block`.
2. **The lookback walk order.** At token t, `rbMode=lookback` starts from 0 at x_t and walks *backward* to x_(t−N+1), so the oldest byte in the window is the last thing the record sees. That is the spec as written ("works backwards from wherever R_A is"). Argon's explanation for why N=64 does worse than N=16 (the oldest byte dominates and recent bytes get washed out) is correctly labelled **Hypothesis**. The obvious control is the same window walked *forward*, ending at x_t.
3. **The 86% / 14% decomposition of the old `scoutScale=0.5` damage** (shared readout vs reverse gradients into the core) is arithmetic on measured paired means, labelled correctly: 0.1057 + 0.0170 = 0.1227.
4. **Hypothesis vs Measured labelling** was followed throughout. "Inconclusive" is never presented as "equivalent."

## What v0.0015 established

- R_B, the partner record, doesn't earn its 3,072 parameters at 1,000 steps (A1 is inconclusive vs B0).
- R_B settles anti-correlated with R_A, at cos ≈ −0.77.
- No partner-record mode beats the plain forward carry, including the lookback window.
- Separating the scout's readout recovers most of the reverse-loss damage, but scoutScale=0 is still best.
- The ternary GRU still leads: by 0.11 nats at 1,000 steps, and by about 0.03 nats at 7,030 steps (inconclusive with 3 seeds).
