#!/usr/bin/env node
// Paired statistical analysis and Markdown table generator for LEARNS v0.0014 ("Signal and Noise").
// Reads results.jsonl and prints every table in EXPERIMENT_REPORT.md:
//   - Per-seed values (seeds 3, 7, 11, 13, 17 for 5-seed studies; 3, 7, 11 for 7,030-step full passes)
//   - Means, tail min/max/mean/range (--tail 5)
//   - Paired Δ with mean, sample standard deviation (s_Δ), and 95% Student-t confidence interval (df = n - 1)
//   - Tier 1 Dual Verdict Rule: 95% CI of paired Δ excludes 0 AND |mean Δ| > max(meanTailRange_A, meanTailRange_B)
// Usage:
//   node tests/tables.js [results.jsonl] [--section=all|tier1|tier2|tier3_lr|tier3_main|tier3_cap|tier3_stream|tier3_long]
'use strict';

const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const fileArg = args.find(a => !a.startsWith('--')) || path.join(__dirname, '..', 'results.jsonl');
const secArg = (args.find(a => a.startsWith('--section=')) || '--section=all').split('=')[1];

if (!fs.existsSync(fileArg)) {
  console.error(`File not found: ${fileArg}`);
  process.exit(1);
}

// Student-t critical values for two-sided 95% CI (alpha = 0.05)
const T_CRIT_95 = {
  1: 12.706204736432095,
  2: 4.302652729911275,
  3: 3.182446305284263,
  4: 2.7764451051977987,
  5: 2.5705818366147395,
  6: 2.4469118511449692,
  7: 2.3646242515927844,
  8: 2.306004135204166,
  9: 2.2621571628540993
};

const rawLines = fs.readFileSync(fileArg, 'utf8').trim().split('\n').filter(Boolean);
const records = [];
for (const line of rawLines) {
  try {
    records.push(JSON.parse(line));
  } catch (e) {}
}

// Index latest record by (config, steps, stream, ngramData, seed)
const byKey = new Map();
for (const r of records) {
  const k = `${r.config}|${r.steps}|${r.stream || 'stride'}|${r.ngramData || ''}|${r.seed}`;
  byKey.set(k, r);
}

function getRec(config, steps, stream, seed, ngramData = '') {
  return byKey.get(`${config}|${steps}|${stream}|${ngramData}|${seed}`) || null;
}

function mean(arr) {
  if (!arr.length) return NaN;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function sampleSD(arr) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  let s = 0;
  for (const x of arr) s += (x - m) * (x - m);
  return Math.sqrt(s / (arr.length - 1));
}

function fmt(x, d = 4) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return Number(x).toFixed(d);
}

function fmtSigned(x, d = 4) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  const v = Number(x);
  return (v >= 0 ? '+' : '') + v.toFixed(d);
}

function computePairedStats(baseRecs, varRecs, field = 'testLoss') {
  const diffs = [];
  const baseTailRanges = [];
  const varTailRanges = [];
  const baseVals = [];
  const varVals = [];

  for (let i = 0; i < baseRecs.length; i++) {
    const b = baseRecs[i], v = varRecs[i];
    if (!b || !v) continue;
    const bv = b[field], vv = v[field];
    if (typeof bv === 'number' && typeof vv === 'number') {
      diffs.push(vv - bv);
      baseVals.push(bv);
      varVals.push(vv);
    }
    if (typeof b.valTailRange === 'number') baseTailRanges.push(b.valTailRange);
    if (typeof v.valTailRange === 'number') varTailRanges.push(v.valTailRange);
  }

  const n = diffs.length;
  if (n === 0) return null;
  const dMean = mean(diffs);
  const dSD = sampleSD(diffs);
  const se = n > 1 ? dSD / Math.sqrt(n) : 0;
  const tCrit = T_CRIT_95[n - 1] || 1.96;
  const ciLo = dMean - tCrit * se;
  const ciHi = dMean + tCrit * se;

  const meanTailBase = baseTailRanges.length ? mean(baseTailRanges) : 0;
  const meanTailVar = varTailRanges.length ? mean(varTailRanges) : 0;
  const maxTailRange = Math.max(meanTailBase, meanTailVar);

  const ciExcludesZero = (ciLo > 0) || (ciHi < 0);
  const exceedsTail = Math.abs(dMean) > maxTailRange;

  let verdict = 'Inconclusive';
  if (ciExcludesZero && exceedsTail) {
    verdict = dMean < 0 ? 'Improves' : 'Degrades';
  }

  return {
    n,
    baseMean: mean(baseVals),
    varMean: mean(varVals),
    dMean,
    dSD,
    ciLo,
    ciHi,
    meanTailBase,
    meanTailVar,
    maxTailRange,
    ciExcludesZero,
    exceedsTail,
    verdict
  };
}

function getSeries(config, steps, stream, seeds, ngramData = '') {
  return seeds.map(s => getRec(config, steps, stream, s, ngramData));
}

function printTier1Table() {
  const seeds = [3, 7, 11, 13, 17];
  const baseCfg = 'L12:arch=learns;dim=32;lr=0.016';
  const rows = [
    { label: '`fpNorm=rescale;fpInject=on;ema=off` *(v0.0014 default)*', cfg: 'L12:arch=learns;dim=32;lr=0.016', isBase: true },
    { label: '`fpNorm=rescale;fpInject=off;ema=off`', cfg: 'L12:arch=learns;dim=32;lr=0.016;fpInject=off' },
    { label: '`fpNorm=rescale;fpInject=on;ema=on`', cfg: 'L12:arch=learns;dim=32;lr=0.016;ema=on' },
    { label: '`fpNorm=rescale;fpInject=off;ema=on`', cfg: 'L12:arch=learns;dim=32;lr=0.016;fpInject=off;ema=on' },
    { label: '`fpNorm=layernorm;fpInject=off` *(v0.0013 default)*', cfg: 'L12:arch=learns;dim=32;lr=0.016;fpInject=off;fpNorm=layernorm' },
    { label: '`fpNorm=layernorm;fpInject=on`', cfg: 'L12:arch=learns;dim=32;lr=0.016;fpNorm=layernorm' },
    { label: '`fpNorm=none;fpInject=on`', cfg: 'L12:arch=learns;dim=32;lr=0.016;fpNorm=none' }
  ];

  const baseRecs = getSeries(baseCfg, 1000, 'stride', seeds);
  console.log('### Table 1: Tier 1 Baseline Selection & Fixed-Point / EMA Candidates (1,000 Steps, `--stream=stride`, 5 Seeds `3, 7, 11, 13, 17`, `--tail 5`)');
  console.log('');
  console.log('| Configuration | Val Mean | Val Seeds (`3 / 7 / 11 / 13 / 17`) | Val Tail Mean | Mean Tail Range | Paired $\\Delta\\text{Val}$ Mean $\\pm s_\\Delta$ | 95% Student-$t$ CI ($df=4$) | Max Tail Range | Tier 1 Verdict (Val) |');
  console.log('|---|---:|---|---:|---:|---:|---|---:|---|');

  for (const r of rows) {
    const recs = getSeries(r.cfg, 1000, 'stride', seeds);
    if (recs.some(x => !x)) continue;
    const valSeeds = recs.map(x => fmt(x.valLoss)).join(' / ');
    const valMean = fmt(mean(recs.map(x => x.valLoss)));
    const tailMean = fmt(mean(recs.map(x => x.valTailMean)));
    const tailRange = fmt(mean(recs.map(x => x.valTailRange)));
    if (r.isBase) {
      console.log(`| ${r.label} | **${valMean}** | ${valSeeds} | ${tailMean} | ${tailRange} | — *(baseline)* | — | ${tailRange} | **Baseline** |`);
    } else {
      const st = computePairedStats(baseRecs, recs, 'valLoss');
      const ciStr = `[${fmtSigned(st.ciLo)}, ${fmtSigned(st.ciHi)}]`;
      console.log(`| ${r.label} | ${valMean} | ${valSeeds} | ${tailMean} | ${tailRange} | ${fmtSigned(st.dMean)} $\\pm$ ${fmt(st.dSD)} | \`${ciStr}\` | ${fmt(st.maxTailRange)} | **${st.verdict}** |`);
    }
  }
  console.log('');
}

function printTier2Tables() {
  const seeds = [3, 7, 11, 13, 17];
  const baseCfg = 'L12:arch=learns;dim=32;lr=0.016';
  const ablations = [
    { label: '`fpNorm=rescale;fpInject=on` *(default)*', comp: 'All 6 components active', cfg: 'L12:arch=learns;dim=32;lr=0.016', isBase: true },
    { label: '`frontierPresent=off`', comp: 'C4: Frontier present readout ($W_{\\text{present}}=0$)', cfg: 'L12:arch=learns;dim=32;lr=0.016;frontierPresent=off' },
    { label: '`helix=off`', comp: 'C5: Helical State disabled ($h_t=P_t$)', cfg: 'L12:arch=learns;dim=32;lr=0.016;helix=off' },
    { label: '`helixRetention=none`', comp: 'C5: Undamped rotation ($\\rho_t \\equiv 1$)', cfg: 'L12:arch=learns;dim=32;lr=0.016;helixRetention=none' },
    { label: '`helixRetention=fixed`', comp: 'C5: Fixed retention ($\\rho \\equiv 0.9$)', cfg: 'L12:arch=learns;dim=32;lr=0.016;helixRetention=fixed' },
    { label: '`thermo=off`', comp: 'C6: Thermodynamic loss ($\\lambda=0$)', cfg: 'L12:arch=learns;dim=32;lr=0.016;thermo=off' },
    { label: '`coreA=off`', comp: 'C2: Temporal Turing Core A ($a_t=0$)', cfg: 'L12:arch=learns;dim=32;lr=0.016;coreA=off' },
    { label: '`coreB=off`', comp: 'C2: Temporal Turing Core B ($b_t=0$)', cfg: 'L12:arch=learns;dim=32;lr=0.016;coreB=off' },
    { label: '`relEmbed=off`', comp: 'C1: Relational Embedding ($W_{\\text{rel}}=0$)', cfg: 'L12:arch=learns;dim=32;lr=0.016;relEmbed=off' },
    { label: '`fpIters=0`', comp: 'C3: Dyadic Fixed Point ($K_A^{(0)}, K_B^{(0)}$ only)', cfg: 'L12:arch=learns;dim=32;lr=0.016;fpIters=0' },
    { label: '`fpInject=off`', comp: 'C3: No input injection in Jacobi loop', cfg: 'L12:arch=learns;dim=32;lr=0.016;fpInject=off' },
    { label: '`fpNorm=none`', comp: 'C3: Contractive Jacobi without norm', cfg: 'L12:arch=learns;dim=32;lr=0.016;fpNorm=none' },
    { label: '`fpNorm=layernorm`', comp: 'C3: LayerNorm inside Jacobi (`fpInject=on`)', cfg: 'L12:arch=learns;dim=32;lr=0.016;fpNorm=layernorm' },
    { label: '`fpNorm=layernorm;fpInject=off`', comp: 'C3: v0.0013 default (`layernorm`, no inject)', cfg: 'L12:arch=learns;dim=32;lr=0.016;fpInject=off;fpNorm=layernorm' },
    { label: '`ema=on`', comp: 'Tier 1: EMA shadow weights ($\\beta=0.99$)', cfg: 'L12:arch=learns;dim=32;lr=0.016;ema=on' }
  ];

  const baseRecs = getSeries(baseCfg, 1000, 'stride', seeds);

  console.log('### Table 2A: Tier 2 Single-Factor Ablation Verdicts — Held-Out Test Loss (1,000 Steps, `--stream=stride`, 5 Seeds `3, 7, 11, 13, 17`, `--tail 5 --final`)');
  console.log('');
  console.log('| Switch / Configuration | Component | Val Mean | Mean Tail Range | Test Mean | Test Seeds (`3 / 7 / 11 / 13 / 17`) | Paired $\\Delta\\text{Test}$ Mean $\\pm s_\\Delta$ | 95% Student-$t$ CI ($df=4$) | Max Tail Range | Tier 1 Verdict |');
  console.log('|---|---|---:|---:|---:|---|---:|---|---:|---|');

  for (const r of ablations) {
    const recs = getSeries(r.cfg, 1000, 'stride', seeds);
    if (recs.some(x => !x)) continue;
    const valMean = fmt(mean(recs.map(x => x.valLoss)));
    const tailRange = fmt(mean(recs.map(x => x.valTailRange)));
    const testMean = fmt(mean(recs.map(x => x.testLoss)));
    const testSeeds = recs.map(x => fmt(x.testLoss)).join(' / ');
    if (r.isBase) {
      console.log(`| ${r.label} | ${r.comp} | ${valMean} | ${tailRange} | **${testMean}** | ${testSeeds} | — *(baseline)* | — | ${tailRange} | **Baseline** |`);
    } else {
      const st = computePairedStats(baseRecs, recs, 'testLoss');
      const ciStr = `[${fmtSigned(st.ciLo)}, ${fmtSigned(st.ciHi)}]`;
      console.log(`| ${r.label} | ${r.comp} | ${valMean} | ${tailRange} | ${testMean} | ${testSeeds} | ${fmtSigned(st.dMean)} $\\pm$ ${fmt(st.dSD)} | \`${ciStr}\` | ${fmt(st.maxTailRange)} | **${st.verdict}** |`);
    }
  }
  console.log('');

  console.log('### Table 2B: Tier 2 Single-Factor Ablation Verdicts — Validation Loss & Diagnostics (1,000 Steps, `--stream=stride`, 5 Seeds `3, 7, 11, 13, 17`)');
  console.log('');
  console.log('| Switch / Configuration | Val Mean | Val Tail Mean | Mean Tail Range | Paired $\\Delta\\text{Val}$ Mean $\\pm s_\\Delta$ | 95% Student-$t$ CI ($df=4$) | Tier 1 Verdict (Val) | Mean `fpResidual` | Mean `spikeCount` | Max `maxBetaGrad` |');
  console.log('|---|---:|---:|---:|---:|---|---|---:|---:|---:|');

  for (const r of ablations) {
    const recs = getSeries(r.cfg, 1000, 'stride', seeds);
    if (recs.some(x => !x)) continue;
    const valMean = fmt(mean(recs.map(x => x.valLoss)));
    const tailMean = fmt(mean(recs.map(x => x.valTailMean)));
    const tailRange = fmt(mean(recs.map(x => x.valTailRange)));
    const fpRes = mean(recs.map(x => x.fpResidual || 0)).toExponential(2);
    const spk = fmt(mean(recs.map(x => x.spikeCount || 0)), 1);
    const maxB = Math.max(...recs.map(x => x.maxBetaGrad || 0)).toExponential(2);
    if (r.isBase) {
      console.log(`| ${r.label} | **${valMean}** | ${tailMean} | ${tailRange} | — *(baseline)* | — | **Baseline** | ${fpRes} | ${spk} | ${maxB} |`);
    } else {
      const st = computePairedStats(baseRecs, recs, 'valLoss');
      const ciStr = `[${fmtSigned(st.ciLo)}, ${fmtSigned(st.ciHi)}]`;
      console.log(`| ${r.label} | ${valMean} | ${tailMean} | ${tailRange} | ${fmtSigned(st.dMean)} $\\pm$ ${fmt(st.dSD)} | \`${ciStr}\` | **${st.verdict}** | ${fpRes} | ${spk} | ${maxB} |`);
    }
  }
  console.log('');
}

function printTier3LRSweep() {
  console.log('### Table 3A: Tier 3 & Tier 4 Validation Learning-Rate Sweeps (`seed=3`, 1,000 Steps, `--stream=stride`, `--tail 5`, Validation Only)');
  console.log('');
  console.log('| Model Family | Candidate `lr` | Config String | Val Final | Val Tail Min | Val Tail Max | Val Tail Mean | Val Tail Range | Selected on Val? |');
  console.log('|---|---:|---|---:|---:|---:|---:|---:|---|');

  const groups = [
    {
      name: 'LEARNS `weights=float32` (`dim=32`)',
      items: [
        { lr: '0.002', cfg: 'L12:arch=learns;dim=32;lr=0.002;weights=float32' },
        { lr: '0.004', cfg: 'L12:arch=learns;dim=32;weights=float32' },
        { lr: '0.008', cfg: 'L12:arch=learns;dim=32;lr=0.008;weights=float32' },
        { lr: '0.016', cfg: 'L12:arch=learns;dim=32;lr=0.016;weights=float32' }
      ]
    },
    {
      name: 'LEARNS `dim=64` (`weights=ternary`)',
      items: [
        { lr: '0.004', cfg: 'L12:arch=learns;dim=64' },
        { lr: '0.008', cfg: 'L12:arch=learns;dim=64;lr=0.008' },
        { lr: '0.016', cfg: 'L12:arch=learns;dim=64;lr=0.016' }
      ]
    },
    {
      name: 'GRU `dim=64` (`weights=ternary`)',
      items: [
        { lr: '0.004', cfg: 'L12:arch=gru;dim=64' },
        { lr: '0.008', cfg: 'L12:arch=gru;dim=64;lr=0.008' },
        { lr: '0.016', cfg: 'L12:arch=gru;dim=64;lr=0.016' }
      ]
    },
    {
      name: 'LEARNS `scoutScale=0` (`dim=32, ternary`)',
      items: [
        { lr: '0.008', cfg: 'L12:arch=learns;dim=32;lr=0.008;scoutScale=0' },
        { lr: '0.016', cfg: 'L12:arch=learns;dim=32;lr=0.016;scoutScale=0' },
        { lr: '0.024', cfg: 'L12:arch=learns;dim=32;lr=0.024;scoutScale=0' }
      ]
    }
  ];

  for (const g of groups) {
    let bestCfg = null, bestVal = Infinity;
    for (const it of g.items) {
      const r = getRec(it.cfg, 1000, 'stride', 3);
      if (r && r.valLoss < bestVal) { bestVal = r.valLoss; bestCfg = it.cfg; }
    }
    for (const it of g.items) {
      const r = getRec(it.cfg, 1000, 'stride', 3);
      if (!r) continue;
      const win = it.cfg === bestCfg;
      console.log(`| ${g.name} | \`${it.lr}\` | \`${r.config}\` | ${win ? '**' + fmt(r.valLoss) + '**' : fmt(r.valLoss)} | ${fmt(r.valTailMin)} | ${fmt(r.valTailMax)} | ${fmt(r.valTailMean)} | ${fmt(r.valTailRange)} | ${win ? '**Yes (Winner)**' : 'No'} |`);
    }
  }
  console.log('');
}

function printTier3MainTable() {
  const seeds = [3, 7, 11, 13, 17];
  const baseCfg = 'L12:arch=learns;dim=32;lr=0.016';
  const baseRecs = getSeries(baseCfg, 1000, 'stride', seeds);

  const arms = [
    { label: 'LEARNS default (`ternary, scoutScale=0.5`)', params: '54,400', stateFloats: '96', cfg: 'L12:arch=learns;dim=32;lr=0.016', isBase: true },
    { label: 'LEARNS `helixState=per_layer` (`ternary`)', params: '54,400', stateFloats: '128', cfg: 'L12:arch=learns;dim=32;lr=0.016;helixState=per_layer' },
    { label: 'LEARNS `weights=float32` (`scoutScale=0.5`)', params: '54,400', stateFloats: '96', cfg: 'L12:arch=learns;dim=32;lr=0.016;weights=float32' },
    { label: 'LEARNS `scoutScale=0` (`ternary`, Tier 4)', params: '54,400', stateFloats: '96', cfg: 'L12:arch=learns;dim=32;lr=0.016;scoutScale=0' },
    { label: 'LEARNS `weights=float32;scoutScale=0` (Tier 3+4)', params: '54,400', stateFloats: '96', cfg: 'L12:arch=learns;dim=32;lr=0.016;weights=float32;scoutScale=0' },
    { label: 'Transformer `ternary` (`dim=32`)', params: '54,656', stateFloats: 'KV(64)', cfg: 'L12:arch=transformer;dim=32;lr=0.016' },
    { label: 'Transformer `float32` (`dim=32`)', params: '54,656', stateFloats: 'KV(64)', cfg: 'L12:arch=transformer;dim=32;lr=0.008;weights=float32' },
    { label: 'GRU `ternary` (`dim=32`)', params: '54,450', stateFloats: '98', cfg: 'L12:arch=gru;dim=32;lr=0.016' },
    { label: 'GRU `float32` (`dim=32`)', params: '54,450', stateFloats: '98', cfg: 'L12:arch=gru;dim=32;lr=0.016;weights=float32' }
  ];

  console.log('### Table 3B: Tier 3 & Tier 4 Architecture & Bottleneck Comparison at `dim=32` (1,000 Steps, `--stream=stride`, 5 Seeds `3, 7, 11, 13, 17`, `--tail 5 --final`)');
  console.log('');
  console.log('| Architecture / Configuration | Params | Recurrent State | Val Mean | Mean Tail Range | Test Mean | Test Seeds (`3 / 7 / 11 / 13 / 17`) | Paired $\\Delta\\text{Test}$ vs LEARNS | 95% Student-$t$ CI ($df=4$) | Tier 1 Verdict |');
  console.log('|---|---:|---:|---:|---:|---:|---|---:|---|---|');

  for (const a of arms) {
    const recs = getSeries(a.cfg, 1000, 'stride', seeds);
    if (recs.some(x => !x)) continue;
    const valMean = fmt(mean(recs.map(x => x.valLoss)));
    const tailRange = fmt(mean(recs.map(x => x.valTailRange)));
    const testMean = fmt(mean(recs.map(x => x.testLoss)));
    const testSeeds = recs.map(x => fmt(x.testLoss)).join(' / ');
    if (a.isBase) {
      console.log(`| ${a.label} | ${a.params} | ${a.stateFloats} | ${valMean} | ${tailRange} | **${testMean}** | ${testSeeds} | — *(baseline)* | — | **Baseline** |`);
    } else {
      const st = computePairedStats(baseRecs, recs, 'testLoss');
      const ciStr = `[${fmtSigned(st.ciLo)}, ${fmtSigned(st.ciHi)}]`;
      console.log(`| ${a.label} | ${a.params} | ${a.stateFloats} | ${valMean} | ${tailRange} | ${testMean} | ${testSeeds} | ${fmtSigned(st.dMean)} $\\pm$ ${fmt(st.dSD)} | \`${ciStr}\` | **${st.verdict}** |`);
    }
  }
  console.log('');
}

function printTier3CapTable() {
  const seeds = [3, 7, 11, 13, 17];
  const baseCfg = 'L12:arch=learns;dim=32;lr=0.016';
  const baseRecs = getSeries(baseCfg, 1000, 'stride', seeds);

  const arms = [
    { label: 'LEARNS `dim=32` (`helixState=shared`)', params: '54,400', state: '96 floats ($R_A:32, R_B:32, h:32$)', cfg: 'L12:arch=learns;dim=32;lr=0.016', refRecs: baseRecs, isBase: true },
    { label: 'LEARNS `dim=32;helixState=per_layer`', params: '54,400', state: '128 floats ($R_A:32, R_B:32, h^{(0)}:32, h^{(1)}:32$)', cfg: 'L12:arch=learns;dim=32;lr=0.016;helixState=per_layer', refRecs: baseRecs },
    { label: 'LEARNS `dim=64` (`lr=0.008`)', params: '164,224', state: '192 floats ($R_A:64, R_B:64, h:64$)', cfg: 'L12:arch=learns;dim=64;lr=0.008', refRecs: baseRecs },
    { label: 'GRU `dim=32` (`weights=ternary, lr=0.016`)', params: '54,450', state: '98 floats ($h^{(0)}:49, h^{(1)}:49$)', cfg: 'L12:arch=gru;dim=32;lr=0.016', refRecs: baseRecs },
    { label: 'GRU `dim=64` (`weights=ternary, lr=0.008`)', params: '165,914', state: '192 floats ($h^{(0)}:96, h^{(1)}:96$)', cfg: 'L12:arch=gru;dim=64;lr=0.008', refRecs: getSeries('L12:arch=learns;dim=64;lr=0.008', 1000, 'stride', seeds), refLabel: 'vs LEARNS `dim=64`' }
  ];

  console.log('### Table 4: Tier 3.2 State Capacity & Width Scaling (`dim=32` vs `dim=64` and `helixState=per_layer`, 1,000 Steps, 5 Seeds `3, 7, 11, 13, 17`)');
  console.log('');
  console.log('| Configuration | Params | Recurrent State | Val Mean | Mean Tail Range | Test Mean | Test Seeds (`3 / 7 / 11 / 13 / 17`) | Paired $\\Delta\\text{Test}$ | 95% Student-$t$ CI ($df=4$) | Tier 1 Verdict |');
  console.log('|---|---:|---|---:|---:|---:|---|---:|---|---|');

  for (const a of arms) {
    const recs = getSeries(a.cfg, 1000, 'stride', seeds);
    if (recs.some(x => !x)) continue;
    const valMean = fmt(mean(recs.map(x => x.valLoss)));
    const tailRange = fmt(mean(recs.map(x => x.valTailRange)));
    const testMean = fmt(mean(recs.map(x => x.testLoss)));
    const testSeeds = recs.map(x => fmt(x.testLoss)).join(' / ');
    if (a.isBase) {
      console.log(`| ${a.label} | ${a.params} | ${a.state} | ${valMean} | ${tailRange} | **${testMean}** | ${testSeeds} | — *(baseline)* | — | **Baseline** |`);
    } else {
      const st = computePairedStats(a.refRecs, recs, 'testLoss');
      const ciStr = `[${fmtSigned(st.ciLo)}, ${fmtSigned(st.ciHi)}]`;
      const note = a.refLabel ? ` *(${a.refLabel})*` : '';
      console.log(`| ${a.label} | ${a.params} | ${a.state} | ${valMean} | ${tailRange} | ${testMean} | ${testSeeds} | ${fmtSigned(st.dMean)} $\\pm$ ${fmt(st.dSD)}${note} | \`${ciStr}\` | **${st.verdict}** |`);
    }
  }
  console.log('');
}

function printTier3StreamTable() {
  const seeds = [3, 7, 11, 13, 17];
  console.log('### Table 5: Tier 3.3 Disentangling `--stream=head` vs `--stream=block` vs `--stream=stride` (150 Steps, 9,600 Bytes Seen, 5 Seeds `3, 7, 11, 13, 17`)');
  console.log('');
  console.log('| Model (`dim=32, lr=0.016`) | Stream Mode | Val Mean | Test Mean | Test Seeds (`3 / 7 / 11 / 13 / 17`) | Mean Tail Range | Paired $\\Delta\\text{Test}$ vs `head` | 95% CI vs `head` ($df=4$) | Paired $\\Delta\\text{Test}$ vs `block` | 95% CI vs `block` ($df=4$) |');
  console.log('|---|---|---:|---:|---|---:|---:|---|---:|---|');

  for (const [mLabel, cfg] of [['LEARNS v0.0014 (`lr=0.016, rescale, inject`)', 'L12:arch=learns;dim=32;lr=0.016'], ['LEARNS v0.0013 (`lr=0.004, layernorm, no-inject`)', 'L12:arch=learns;dim=32;fpInject=off;fpNorm=layernorm'], ['GRU (`weights=ternary, lr=0.016`)', 'L12:arch=gru;dim=32;lr=0.016']]) {
    const headRecs = getSeries(cfg, 150, 'head', seeds);
    const blockRecs = getSeries(cfg, 150, 'block', seeds);
    const strideRecs = getSeries(cfg, 150, 'stride', seeds);
    if (headRecs.some(x => !x) || blockRecs.some(x => !x) || strideRecs.some(x => !x)) continue;

    for (const [sMode, recs] of [['`head` *(bytes 0..9,600)*', headRecs], ['`block` *(1 random 9,600B block)*', blockRecs], ['`stride` *(150 strided chunks)*', strideRecs]]) {
      const valMean = fmt(mean(recs.map(x => x.valLoss)));
      const testMean = fmt(mean(recs.map(x => x.testLoss)));
      const testSeeds = recs.map(x => fmt(x.testLoss)).join(' / ');
      const tailRange = fmt(mean(recs.map(x => x.valTailRange)));

      let dHead = '—', ciHead = '—', dBlock = '—', ciBlock = '—';
      if (recs !== headRecs) {
        const stH = computePairedStats(headRecs, recs, 'testLoss');
        dHead = `${fmtSigned(stH.dMean)} $\\pm$ ${fmt(stH.dSD)}`;
        ciHead = `\`[${fmtSigned(stH.ciLo)}, ${fmtSigned(stH.ciHi)}]\``;
      }
      if (recs === strideRecs) {
        const stB = computePairedStats(blockRecs, recs, 'testLoss');
        dBlock = `${fmtSigned(stB.dMean)} $\\pm$ ${fmt(stB.dSD)}`;
        ciBlock = `\`[${fmtSigned(stB.ciLo)}, ${fmtSigned(stB.ciHi)}]\``;
      }
      console.log(`| ${mLabel} | ${sMode} | ${valMean} | **${testMean}** | ${testSeeds} | ${tailRange} | ${dHead} | ${ciHead} | ${dBlock} | ${ciBlock} |`);
    }
  }
  console.log('');
}

function printTier3LongTable() {
  const seeds = [3, 7, 11];
  const baseCfg = 'L12:arch=learns;dim=32;lr=0.016';
  const baseRecs = getSeries(baseCfg, 7030, 'stride', seeds);
  const scoutRecs = getSeries('L12:arch=learns;dim=32;lr=0.016;scoutScale=0', 7030, 'stride', seeds);

  const arms = [
    { label: 'LEARNS v0.0014 default (`fpNorm=rescale;fpInject=on`)', cfg: 'L12:arch=learns;dim=32;lr=0.016', isBase: true },
    { label: 'LEARNS v0.0013 legacy (`fpNorm=layernorm;fpInject=off`)', cfg: 'L12:arch=learns;dim=32;lr=0.016;fpInject=off;fpNorm=layernorm' },
    { label: 'LEARNS `relEmbed=off` (`fpNorm=layernorm;fpInject=off`)', cfg: 'L12:arch=learns;dim=32;lr=0.016;relEmbed=off;fpInject=off;fpNorm=layernorm' },
    { label: 'LEARNS `weights=float32` (`lr=0.016`)', cfg: 'L12:arch=learns;dim=32;lr=0.016;weights=float32' },
    { label: 'LEARNS `scoutScale=0` (`ternary, lr=0.016`, Tier 4)', cfg: 'L12:arch=learns;dim=32;lr=0.016;scoutScale=0' },
    { label: 'LEARNS `weights=float32;scoutScale=0` (`lr=0.016`, Tier 3+4)', cfg: 'L12:arch=learns;dim=32;lr=0.016;weights=float32;scoutScale=0' },
    { label: 'Transformer `ternary` (`dim=32, lr=0.016`)', cfg: 'L12:arch=transformer;dim=32;lr=0.016' },
    { label: 'Transformer `float32` (`dim=32, lr=0.008`)', cfg: 'L12:arch=transformer;dim=32;lr=0.008;weights=float32' },
    { label: 'GRU `ternary` (`dim=32, lr=0.016`)', cfg: 'L12:arch=gru;dim=32;lr=0.016' },
    { label: 'GRU `ternary` vs LEARNS `scoutScale=0` *(paired)*', cfg: 'L12:arch=gru;dim=32;lr=0.016', refRecs: scoutRecs, refLabel: 'vs `scoutScale=0`' },
    { label: 'GRU `float32` (`dim=32, lr=0.016`)', cfg: 'L12:arch=gru;dim=32;lr=0.016;weights=float32' },
    { label: 'GRU `float32` (`dim=32, lr=0.008`)', cfg: 'L12:arch=gru;dim=32;lr=0.008;weights=float32' }
  ];

  console.log('### Table 6A: Tier 3.4 & Tier 4 Full-Book Training Pass (`7,030` Steps = 1.0 Epoch, `--stream=stride`, 3 Seeds `3, 7, 11`, `--tail 5 --final`)');
  console.log('');
  console.log('| Architecture / Configuration | Val Mean | Mean Tail Range | Test Mean | Test Seeds (`3 / 7 / 11`) | Paired $\\Delta\\text{Test}$ vs LEARNS v0.0014 | 95% Student-$t$ CI ($df=2$) | Tier 1 Verdict |');
  console.log('|---|---:|---:|---:|---|---:|---|---|');

  for (const a of arms) {
    const recs = getSeries(a.cfg, 7030, 'stride', seeds);
    if (recs.some(x => !x)) continue;
    const valMean = fmt(mean(recs.map(x => x.valLoss)));
    const tailRange = fmt(mean(recs.map(x => x.valTailRange)));
    const testMean = fmt(mean(recs.map(x => x.testLoss)));
    const testSeeds = recs.map(x => fmt(x.testLoss)).join(' / ');
    const ref = a.refRecs || baseRecs;
    if (a.isBase || ref.some(x => !x)) {
      console.log(`| ${a.label} | ${valMean} | ${tailRange} | **${testMean}** | ${testSeeds} | — *(baseline)* | — | **Baseline** |`);
    } else {
      const st = computePairedStats(ref, recs, 'testLoss');
      const ciStr = `[${fmtSigned(st.ciLo)}, ${fmtSigned(st.ciHi)}]`;
      const note = a.refLabel ? ` *(${a.refLabel})*` : '';
      console.log(`| ${a.label} | ${valMean} | ${tailRange} | ${testMean} | ${testSeeds} | ${fmtSigned(st.dMean)} $\\pm$ ${fmt(st.dSD)}${note} | \`${ciStr}\` | **${st.verdict}** |`);
    }
  }
  console.log('');

  const checkpoints = [1000, 2000, 3000, 4000, 5000, 6000, 7030];
  console.log('### Table 6B: Full-Pass Validation Learning Curves (3-Seed Mean Validation Loss in nats/byte across Checkpoints `1,000 .. 7,030`)');
  console.log('');
  console.log('| Architecture / Configuration | Step 1,000 | Step 2,000 | Step 3,000 | Step 4,000 | Step 5,000 | Step 6,000 | Step 7,030 (Final) |');
  console.log('|---|---:|---:|---:|---:|---:|---:|---:|');

  for (const a of arms) {
    if (a.refLabel) continue;
    const recs = getSeries(a.cfg, 7030, 'stride', seeds);
    if (recs.some(x => !x || !x.valCurve)) continue;
    const cols = checkpoints.map(cp => {
      const vals = recs.map(r => {
        const pt = r.valCurve.find(p => p.step === cp);
        return pt ? pt.valLoss : NaN;
      });
      return fmt(mean(vals));
    });
    console.log(`| ${a.label} | ${cols.slice(0, 6).join(' | ')} | **${cols[6]}** |`);
  }
  console.log('');
}

if (secArg === 'all' || secArg === 'tier1') printTier1Table();
if (secArg === 'all' || secArg === 'tier2') printTier2Tables();
if (secArg === 'all' || secArg === 'tier3_lr') printTier3LRSweep();
if (secArg === 'all' || secArg === 'tier3_main') printTier3MainTable();
if (secArg === 'all' || secArg === 'tier3_cap') printTier3CapTable();
if (secArg === 'all' || secArg === 'tier3_stream') printTier3StreamTable();
if (secArg === 'all' || secArg === 'tier3_long') printTier3LongTable();