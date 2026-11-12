#!/usr/bin/env node
/**
 * Recomputes all LEARNS v0.0016 ("Rhythm") report tables from results.jsonl
 * and loom-corpus-meta.json.
 * Marks reused runs (carrying reusedFrom + original codeHash) with †.
 */
'use strict';
const fs = require('fs');
const path = require('path');

const rootDir = path.join(__dirname, '..');
const resultsPath = fs.existsSync(path.join(rootDir, 'results.jsonl'))
  ? path.join(rootDir, 'results.jsonl')
  : path.resolve('results.jsonl');
const metaPath = path.join(rootDir, 'loom-corpus-meta.json');

const records = fs.readFileSync(resultsPath, 'utf8')
  .split('\n')
  .filter(l => l.trim())
  .map(l => JSON.parse(l));
const meta = fs.existsSync(metaPath) ? JSON.parse(fs.readFileSync(metaPath, 'utf8')) : null;

function mean(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : NaN;
}
function sd(arr) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, x) => s + (x - m) * (x - m), 0) / (arr.length - 1));
}
// Two-tailed t critical value for df=4 (95% CI): 2.776445
function tCrit95(df) {
  const table = { 1: 12.706, 2: 4.3027, 3: 3.1824, 4: 2.776445, 5: 2.5706 };
  return table[df] || 1.96;
}
function pairedStats(candVals, baseVals, noiseFloor = 0.015) {
  const n = Math.min(candVals.length, baseVals.length);
  const diffs = [];
  let wins = 0;
  for (let i = 0; i < n; i++) {
    const d = candVals[i] - baseVals[i];
    diffs.push(d);
    if (d < 0) wins++;
  }
  const m = mean(diffs);
  const s = sd(diffs);
  const se = n > 1 ? s / Math.sqrt(n) : 0;
  const hw = n > 1 ? tCrit95(n - 1) * se : 0;
  const ciLo = m - hw, ciHi = m + hw;
  let verdict = 'inconclusive';
  if (ciHi < 0 && Math.abs(m) > noiseFloor) verdict = 'WIN';
  else if (ciLo > 0 && Math.abs(m) > noiseFloor) verdict = 'LOSS';
  return { n, meanDiff: m, sdDiff: s, ciLo, ciHi, wins, verdict };
}
function fmt(x, d = 4) {
  return (x === null || x === undefined || Number.isNaN(x)) ? '—' : Number(x).toFixed(d);
}
function fmtSigned(x, d = 4) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return (x >= 0 ? '+' : '') + Number(x).toFixed(d);
}

const out = [];

// Table 1: Tier 1 .loom Corpus & Leak Check Summary
if (meta) {
  out.push('### Table 1: Tier 1 Procedural `.loom` Corpus & Crew Score Leak Check (`seed=20261111`)');
  out.push('');
  out.push('| Split / Crew Score | Author / Source | Channels / Scores | Total Bytes | `step` Bytes | `pitch` Bytes | `other` Bytes | Longest Common Substring vs Train | 16-Gram Unique Overlap |');
  out.push('| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |');
  const cs = meta.corpusStats;
  out.push(`| \`loom-train.txt\` | Synth Radio (110 ch) | 110 ch / ${cs.trainScores} scores | ${cs.trainRoleBytes.total} | ${cs.trainRoleBytes.step} | ${cs.trainRoleBytes.pitch} | ${cs.trainRoleBytes.other} | — | — |`);
  out.push(`| \`loom-val.txt\` | Synth Radio (10 ch) | 10 ch / ${cs.valScores} scores | ${cs.valRoleBytes.total} | ${cs.valRoleBytes.step} | ${cs.valRoleBytes.pitch} | ${cs.valRoleBytes.other} | — | — |`);
  out.push(`| \`loom-test-unseen\` | Synth Radio (26 ch) | 26 ch / ${cs.unseenScores} scores | ${cs.unseenRoleBytes.total} | ${cs.unseenRoleBytes.step} | ${cs.unseenRoleBytes.pitch} | ${cs.unseenRoleBytes.other} | — | — |`);
  out.push(`| \`loom-test-crew\` (all 8) | Studio Crew | 8 scores | ${cs.crewRoleBytes.total} | ${cs.crewRoleBytes.step} | ${cs.crewRoleBytes.pitch} | ${cs.crewRoleBytes.other} | — | — |`);
  for (const c of meta.crewScores) {
    out.push(`| Crew: \`${c.id}\` | ${c.author} (${c.bars} bars, ${c.bpm} BPM) | 1 (${c.woven} woven) | ${c.bytes} | ${c.roleBytes.step} | ${c.roleBytes.pitch} | ${c.roleBytes.other} | ${c.leak.lcsLength} B (\`${c.leak.lcsString.slice(0, 28)}...\`) | ${(c.leak.uniqueOverlapFrac * 100).toFixed(2)}% (${c.leak.matchedUnique16Grams}/${c.leak.unique16Grams}) |`);
  }
  out.push('');
}

// Table 2: Tier 2 Seed-3 Validation LR Sweep on .loom
out.push('### Table 2: Tier 2 Validation Learning-Rate Sweep on `.loom` (`seed=3`, 1,000 Steps)');
out.push('');
out.push('| Model | Weights | Params | `lr=0.004` Val | `lr=0.008` Val | `lr=0.016` Val | `lr=0.024` Val | Selected LR |');
out.push('| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |');
const sweepSpecs = [
  { label: 'GRU', arch: 'gru', weights: 'float32', lrs: [0.008, 0.016, 0.024] },
  { label: 'GRU', arch: 'gru', weights: 'ternary', lrs: [0.008, 0.016, 0.024] },
  { label: 'LEARNS (B0)', arch: 'learns', weights: 'float32', lrs: [0.004, 0.008, 0.016] },
  { label: 'LEARNS (B0)', arch: 'learns', weights: 'ternary', lrs: [0.008, 0.016, 0.024] },
  { label: 'Transformer', arch: 'transformer', weights: 'float32', lrs: [0.004, 0.008, 0.016] },
  { label: 'Transformer', arch: 'transformer', weights: 'ternary', lrs: [0.004, 0.008, 0.016] }
];
for (const sp of sweepSpecs) {
  const vals = {};
  let pCount = '—';
  let bestLR = null, bestV = Infinity;
  for (const lr of [0.004, 0.008, 0.016, 0.024]) {
    const rec = records.find(r =>
      r.task === 'loom' && r.arch === sp.arch && r.seed === 3 && r.steps === 1000 &&
      (sp.weights === 'float32' ? r.config.includes('weights=float32') : !r.config.includes('weights=float32')) &&
      (lr === 0.004 ? (!r.config.includes('lr=') || r.config.includes('lr=0.004')) : r.config.includes(`lr=${lr}`)) &&
      !r.config.includes('helix=') && !r.config.includes('recordB=off') && !r.config.includes('frontierPresent=off')
    );
    if (rec) {
      vals[lr] = rec.valLoss;
      pCount = rec.paramCount;
      if (rec.valLoss < bestV) { bestV = rec.valLoss; bestLR = lr; }
    }
  }
  out.push(`| ${sp.label} | ${sp.weights} | ${pCount} | ${fmt(vals[0.004])} | ${fmt(vals[0.008])} | ${fmt(vals[0.016])} | ${fmt(vals[0.024])} | \`${bestLR}\` |`);
}
out.push('');

// Table 3: Tier 2 Main 5-Seed .loom Benchmark (Validation, Unseen Genres, Crew Scores + Paired Verdicts vs LEARNS B0 Ternary)
out.push('### Table 3: Tier 2 `.loom` Benchmark — 5-Seed Overall & Byte-Role Losses (`seeds=3,7,11,13,17`, 1,000 Steps)');
out.push('');
out.push('| Model / Baseline | Params | Val Mean±SD | Val Tail Range | Unseen Genres Mean±SD | Unseen `step` / `pitch` / `other` | Crew Scores Mean±SD | Crew `step` / `pitch` / `other` | Unseen Δ vs B0 Ternary (95% CI, Verdict) |');
out.push('| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |');

const seeds = [3, 7, 11, 13, 17];
function getLoom5(tag) {
  return seeds.map(s => records.find(r => r.tag === tag && r.seed === s)).filter(Boolean);
}
const b0TernaryRuns = getLoom5('v16_t2_learns_ternary');
const b0Unseen = b0TernaryRuns.map(r => r.unseenTestLoss);

const t2Groups = [
  { label: 'GRU (float32, `lr=0.016`)', tag: 'v16_t2_gru_float32' },
  { label: 'GRU (ternary, `lr=0.016`)', tag: 'v16_t2_gru_ternary' },
  { label: 'LEARNS B0 (float32, `lr=0.008`)', tag: 'v16_t2_learns_float32' },
  { label: 'LEARNS B0 (ternary, `lr=0.016`)', tag: 'v16_t2_learns_ternary', isBase: true },
  { label: 'Transformer (float32, `lr=0.004`)', tag: 'v16_t2_transformer_float32' },
  { label: 'Transformer (ternary, `lr=0.008`)', tag: 'v16_t2_transformer_ternary' }
];

for (const g of t2Groups) {
  const rs = getLoom5(g.tag);
  if (!rs.length) continue;
  const valArr = rs.map(r => r.valLoss);
  const trArr = rs.map(r => r.valTailRange);
  const unArr = rs.map(r => r.unseenTestLoss);
  const crArr = rs.map(r => r.crewTestLoss);
  const uStep = mean(rs.map(r => r.unseenRoleLoss.step));
  const uPitch = mean(rs.map(r => r.unseenRoleLoss.pitch));
  const uOther = mean(rs.map(r => r.unseenRoleLoss.other));
  const cStep = mean(rs.map(r => r.crewRoleLoss.step));
  const cPitch = mean(rs.map(r => r.crewRoleLoss.pitch));
  const cOther = mean(rs.map(r => r.crewRoleLoss.other));
  let cmpStr = '— (baseline)';
  if (!g.isBase && rs.length === 5 && b0Unseen.length === 5) {
    const st = pairedStats(unArr, b0Unseen, 0.010);
    cmpStr = `${fmtSigned(st.meanDiff)} ([${fmtSigned(st.ciLo)}, ${fmtSigned(st.ciHi)}], ${st.wins}/5, **${st.verdict}**)`;
  }
  out.push(`| ${g.label} | ${rs[0].paramCount} | ${fmt(mean(valArr))}±${fmt(sd(valArr))} | ${fmt(mean(trArr))} | ${fmt(mean(unArr))}±${fmt(sd(unArr))} | ${fmt(uStep)} / ${fmt(uPitch)} / ${fmt(uOther)} | ${fmt(mean(crArr))}±${fmt(sd(crArr))} | ${fmt(cStep)} / ${fmt(cPitch)} / ${fmt(cOther)} | ${cmpStr} |`);
}

// Add N-gram rows to Table 3
for (const ng of ['5gram', 'trigram', 'bigram', 'unigram']) {
  for (const mode of ['seen', 'all']) {
    const r = records.find(x => x.task === 'loom' && x.arch === ng && x.ngramData === mode);
    if (!r) continue;
    const uR = r.unseenRoleLoss || {};
    const cR = r.crewRoleLoss || {};
    out.push(`| ${ng} (\`ngramData=${mode}\`) | ${r.paramCount} | ${fmt(r.valLoss)} | 0.0000 | ${fmt(r.unseenTestLoss)} | ${fmt(uR.step)} / ${fmt(uR.pitch)} / ${fmt(uR.other)} | ${fmt(r.crewTestLoss)} | ${fmt(cR.step)} / ${fmt(cR.pitch)} / ${fmt(cR.other)} | — |`);
  }
}
out.push('');

// Table 4: Per-Crew-Score Breakdown (8 Crew Scores, 5-Seed Mean Loss)
out.push('### Table 4: Tier 2 Per-Score Loss on the 8 Comment-Stripped Crew Scores (5-Seed Mean Nats/Byte)');
out.push('');
const crewIds = ['peepers', 'hollow', 'hollow-2', 'threshold', 'understory', 'understory-morning', 'understory-rest', 'understory-trouble'];
out.push('| Model | ' + crewIds.map(id => `\`${id}\``).join(' | ') + ' | Unweighted 8-Score Mean | Byte-Weighted Crew Loss |');
out.push('| :--- | ' + crewIds.map(() => ':---:').join(' | ') + ' | :---: | :---: |');
for (const g of t2Groups) {
  const rs = getLoom5(g.tag);
  if (!rs.length) continue;
  const cells = crewIds.map(id => fmt(mean(rs.map(r => r.crewByScore[id].loss))));
  out.push(`| ${g.label} | ${cells.join(' | ')} | ${fmt(mean(rs.map(r => r.crewScoreMeanLoss)))} | ${fmt(mean(rs.map(r => r.crewTestLoss)))} |`);
}
out.push('');

// Table 5: Per-Channel Breakdown across the 26 Held-Out Unseen Genre Channels (5-Seed Mean Loss)
out.push('### Table 5: Tier 2 Per-Channel Loss across the 26 Held-Out Unseen-Genre Channels (5-Seed Mean Nats/Byte)');
out.push('');
out.push('| Held-Out Channel (`id`) | Cluster | Bytes | GRU (float32) | GRU (ternary) | LEARNS B0 (float32) | LEARNS B0 (ternary) | Transformer (float32) | Transformer (ternary) |');
out.push('| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |');
if (b0TernaryRuns.length > 0 && b0TernaryRuns[0].unseenByChannel) {
  const chKeys = Object.keys(b0TernaryRuns[0].unseenByChannel);
  for (const chId of chKeys) {
    const chMeta = meta && meta.unseenTestChannels ? meta.unseenTestChannels.find(c => c.id === chId) : null;
    const clName = chMeta ? chMeta.cluster : '—';
    const bCount = b0TernaryRuns[0].unseenByChannel[chId].bytes;
    const vals = t2Groups.map(g => {
      const rs = getLoom5(g.tag);
      return fmt(mean(rs.map(r => r.unseenByChannel[chId].loss)));
    });
    out.push(`| \`${chId}\` | ${clName} | ${bCount} | ${vals.join(' | ')} |`);
  }
}
out.push('');

// Table 6: Tier 3 Component & Helix Ablations on .loom
out.push('### Table 6: Tier 3 `.loom` Component & Helix Ablations vs LEARNS B0 Ternary (`lr=0.016`, 1,000 Steps)');
out.push('');
out.push('| Configuration | Seeds | Val Loss | Unseen Genres Loss | Unseen `step` / `pitch` / `other` | Crew Loss | Crew `step` / `pitch` / `other` | Unseen Δ vs B0 (Paired across matched seeds) |');
out.push('| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |');
const t3Groups = [
  { label: 'B0: `learns` default (`helix=on`)', tag: 'v16_t2_learns_ternary', isBase: true },
  { label: 'Ablation: `helix=off`', tag: 'v16_t3_helix_off' },
  { label: 'Ablation: `helixRotation=none`', tag: 'v16_t3_helixRot_none' },
  { label: 'Ablation: `helixRetention=fixed`', tag: 'v16_t3_helixRet_fixed' },
  { label: 'Ablation: `helixRetention=none`', tag: 'v16_t3_helixRet_none' },
  { label: 'Ablation: `frontierPresent=off`', tag: 'v16_t3_frontierPresent_off' },
  { label: 'Ablation: `recordB=off`', tag: 'v16_t3_recordB_off' }
];
for (const g of t3Groups) {
  const rs = records.filter(r => r.tag === g.tag && r.testLoss !== null).sort((a, b) => a.seed - b.seed);
  if (!rs.length) continue;
  const matchedBase = rs.map(r => b0TernaryRuns.find(b => b.seed === r.seed)).filter(Boolean);
  const uStep = mean(rs.map(r => r.unseenRoleLoss.step));
  const uPitch = mean(rs.map(r => r.unseenRoleLoss.pitch));
  const uOther = mean(rs.map(r => r.unseenRoleLoss.other));
  const cStep = mean(rs.map(r => r.crewRoleLoss.step));
  const cPitch = mean(rs.map(r => r.crewRoleLoss.pitch));
  const cOther = mean(rs.map(r => r.crewRoleLoss.other));
  let deltaStr = '— (baseline)';
  if (!g.isBase && matchedBase.length === rs.length) {
    const diffs = rs.map((r, i) => r.unseenTestLoss - matchedBase[i].unseenTestLoss);
    const stepDiffs = rs.map((r, i) => r.unseenRoleLoss.step - matchedBase[i].unseenRoleLoss.step);
    const pitchDiffs = rs.map((r, i) => r.unseenRoleLoss.pitch - matchedBase[i].unseenRoleLoss.pitch);
    deltaStr = `${fmtSigned(mean(diffs))} (step ${fmtSigned(mean(stepDiffs))}, pitch ${fmtSigned(mean(pitchDiffs))})`;
  }
  out.push(`| ${g.label} | ${rs.length} | ${fmt(mean(rs.map(r => r.valLoss)))} | ${fmt(mean(rs.map(r => r.unseenTestLoss)))} | ${fmt(uStep)} / ${fmt(uPitch)} / ${fmt(uOther)} | ${fmt(mean(rs.map(r => r.crewTestLoss)))} | ${fmt(cStep)} / ${fmt(cPitch)} / ${fmt(cOther)} | ${deltaStr} |`);
}
out.push('');

// Table 7: Tier 4 Look-Back Walk-Order Control on Sherlock Prose (corpus.txt, 5 seeds)
out.push('### Table 7: Tier 4 Look-Back Walk-Order Control on Sherlock Prose (`corpus.txt`, 5 Seeds `3,7,11,13,17`, 1,000 Steps)');
out.push('');
out.push('| Prose Arm (`corpus.txt`) | Walk Order over $x_{t-N+1}\\dots x_t$ | Val Mean±SD | Test Mean±SD | Δ Test vs B0 (95% CI, Wins, Verdict) | Δ Test vs `lookback` Same $N$ (95% CI, Verdict) |');
out.push('| :--- | :--- | :---: | :---: | :--- | :--- |');

function getProse5(tagOrCfg) {
  return seeds.map(s => {
    const fresh = records.find(r => (!r.task || r.task === 'prose') && r.tag === tagOrCfg.tag && r.seed === s && r.testLoss !== null);
    if (fresh) return fresh;
    return records.find(r => (!r.task || r.task === 'prose') && r.tag === tagOrCfg.fallbackTag && r.seed === s && r.testLoss !== null);
  }).filter(Boolean);
}

const pB0 = getProse5({ tag: 'v16_t4_repro_B0', fallbackTag: 'v15_B0_default' });
const pLB16 = getProse5({ tag: 'v16_t4_repro_lookback16', fallbackTag: 'v15_A3_lookback16' });
const pW16 = getProse5({ tag: 'v16_t4_window16', fallbackTag: 'v16_t4_window16' });
const pLB64 = getProse5({ tag: 'v16_t4_repro_lookback64', fallbackTag: 'v15_A3_lookback64' });
const pW64 = getProse5({ tag: 'v16_t4_window64', fallbackTag: 'v16_t4_window64' });

const t4Rows = [
  { label: 'B0: `rbMode=causal` (`scoutScale=0`) †', walk: 'Causal carry ($\\mathbf{R}_{B,t}$ from $\\mathbf{R}_{B,t-1}$ + $\\mathbf{K}_{B,t}$)', runs: pB0, isBase: true },
  { label: 'A3-16: `rbMode=lookback;lookback=16` †', walk: 'Backward ($x_t \\to x_{t-15}$, oldest last)', runs: pLB16 },
  { label: 'T4-W16: `rbMode=window;lookback=16`', walk: 'Forward ($x_{t-15} \\to x_t$, newest last)', runs: pW16, sameN: pLB16 },
  { label: 'A3-64: `rbMode=lookback;lookback=64` †', walk: 'Backward ($x_t \\to x_{t-63}$, oldest last)', runs: pLB64 },
  { label: 'T4-W64: `rbMode=window;lookback=64`', walk: 'Forward ($x_{t-63} \\to x_t$, newest last)', runs: pW64, sameN: pLB64 }
];

const pB0Test = pB0.map(r => r.testLoss);
for (const row of t4Rows) {
  const rs = row.runs;
  if (rs.length < 5) continue;
  const vArr = rs.map(r => r.valLoss);
  const tArr = rs.map(r => r.testLoss);
  let vsB0 = '— (baseline)';
  if (!row.isBase) {
    const st = pairedStats(tArr, pB0Test, 0.015);
    vsB0 = `${fmtSigned(st.meanDiff)} ([${fmtSigned(st.ciLo)}, ${fmtSigned(st.ciHi)}], ${st.wins}/5, **${st.verdict}**)`;
  }
  let vsLB = '—';
  if (row.sameN && row.sameN.length === 5) {
    const stLB = pairedStats(tArr, row.sameN.map(r => r.testLoss), 0.015);
    vsLB = `${fmtSigned(stLB.meanDiff)} ([${fmtSigned(stLB.ciLo)}, ${fmtSigned(stLB.ciHi)}], ${stLB.wins}/5, **${stLB.verdict}**)`;
  }
  out.push(`| ${row.label} | ${row.walk} | ${fmt(mean(vArr))}±${fmt(sd(vArr))} | ${fmt(mean(tArr))}±${fmt(sd(tArr))} | ${vsB0} | ${vsLB} |`);
}
out.push('');
out.push('*(† Rows marked with † reuse seeds 7, 11, 13, 17 from v0.0015 (`codeHash=0x626fd26c`, `reusedFrom="v0.0015"`) after verifying seed 3 is bit-for-bit identical under v0.0016 `codeHash=0x9c28cae9`.)*');

console.log(out.join('\n'));