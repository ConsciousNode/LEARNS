#!/usr/bin/env node
// Paired statistical analysis and Markdown table generator for LEARNS v0.0015 ("The Partner Record").
// Reads results.jsonl and prints every table in EXPERIMENT_REPORT.md:
//   - Per-seed values (seeds 3, 7, 11, 13, 17 for 5-seed studies; 3, 7, 11 for 7,030-step full passes)
//   - Means, tail min/max/mean/range (--tail 5)
//   - Paired Δ with mean, sample standard deviation (s_Δ), and 95% Student-t confidence interval (df = n - 1)
//   - Tier 1 Dual Verdict Rule: 95% CI of paired Δ excludes 0 AND |mean Δ| > max(meanTailRange_A, meanTailRange_B)
//   - Marks any row containing reused records (carrying a reusedFrom field) with †
// Usage:
//   node tests/tables.js [results.jsonl] [--section=all|tier2|tier2_diag|tier2_traj|tier2_lr|tier3]
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

// Separate indices for fresh v0.0015 records (!r.reusedFrom) and reused prior-release records (r.reusedFrom)
const byFreshKey = new Map();
const byReusedKey = new Map();

for (const r of records) {
  const k = `${r.config}|${r.steps}|${r.stream || 'stride'}|${r.ngramData || ''}|${r.seed}`;
  if (r.reusedFrom) {
    byReusedKey.set(k, r);
    byReusedKey.set(`${r.codeHash}|${k}`, r);
  } else {
    byFreshKey.set(k, r);
  }
}

function getFreshRec(config, steps, stream, seed, ngramData = '') {
  return byFreshKey.get(`${config}|${steps}|${stream}|${ngramData}|${seed}`) || null;
}

function getReusedRec(config, steps, stream, seed, codeHash = null, ngramData = '') {
  const k = `${config}|${steps}|${stream}|${ngramData}|${seed}`;
  if (codeHash) return byReusedKey.get(`${codeHash}|${k}`) || null;
  return byReusedKey.get(k) || null;
}

function getFreshSeries(config, steps, stream, seeds, ngramData = '') {
  return seeds.map(s => getFreshRec(config, steps, stream, s, ngramData));
}

function getReusedSeries(config, steps, stream, seeds, codeHash = null, ngramData = '') {
  return seeds.map(s => getReusedRec(config, steps, stream, s, codeHash, ngramData));
}

function markLabelIfReused(label, recs, refRecs = null) {
  const hasReused = (recs && recs.some(r => r && r.reusedFrom)) || (refRecs && refRecs.some(r => r && r.reusedFrom));
  if (hasReused && !label.includes('†')) return `${label} †`;
  return label;
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

function hasMultiTail(recs) {
  return recs && recs.every(r => r && Array.isArray(r.valTail) && r.valTail.length > 1);
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
    if (typeof b.valTailRange === 'number' && Array.isArray(b.valTail) && b.valTail.length > 1) baseTailRanges.push(b.valTailRange);
    if (typeof v.valTailRange === 'number' && Array.isArray(v.valTail) && v.valTail.length > 1) varTailRanges.push(v.valTailRange);
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

function printTier2MainTable() {
  const seeds = [3, 7, 11, 13, 17];
  const b0Recs = getFreshSeries('L12:arch=learns;dim=32;lr=0.016', 1000, 'stride', seeds);
  const v14DefRecs = getReusedSeries('L12:arch=learns;dim=32;lr=0.016', 1000, 'stride', seeds, '0x4048aacb');

  const rows = [
    {
      label: '**B0**: `scoutScale=0;rbMode=causal` *(v0.0015 default)*',
      role: 'Forward causal carry $R_B$, no reverse-LM loss',
      recs: b0Recs,
      isBase: true
    },
    {
      label: '**A1**: `recordB=off` (`scoutScale=0`)',
      role: 'Disable $R_B$ ($W_{\\text{rec},B}, W_{\\text{cross},A}, W_{\\text{cross},B}$ off)',
      recs: getFreshSeries('L12:arch=learns;dim=32;lr=0.016;recordB=off', 1000, 'stride', seeds)
    },
    {
      label: '**A2**: `rbMode=prevrev` (`scoutScale=0`)',
      role: 'Seed $R_B$ at chunk start from reversed prev chunk',
      recs: getFreshSeries('L12:arch=learns;dim=32;lr=0.016;rbMode=prevrev', 1000, 'stride', seeds)
    },
    {
      label: '**A3a**: `rbMode=lookback;lookback=16` (`scoutScale=0`)',
      role: 'Rolling backward recurrence over $x_t \\dots x_{t-15}$',
      recs: getFreshSeries('L12:arch=learns;dim=32;lr=0.016;rbMode=lookback;lookback=16', 1000, 'stride', seeds)
    },
    {
      label: '**A3b**: `rbMode=lookback;lookback=64` (`scoutScale=0`)',
      role: 'Rolling backward recurrence over $x_t \\dots x_{t-63}$',
      recs: getFreshSeries('L12:arch=learns;dim=32;lr=0.016;rbMode=lookback;lookback=64', 1000, 'stride', seeds)
    },
    {
      label: '**A4**: `scoutScale=0.5;scoutHead=separate`',
      role: 'Reverse-LM scout on dedicated $W_{\\text{frontier,scout}}, W_{\\text{present,scout}}$',
      recs: getFreshSeries('L12:arch=learns;dim=32;lr=0.016;scoutScale=0.5;scoutHead=separate', 1000, 'stride', seeds)
    },
    {
      label: 'v0.0014 default (`scoutScale=0.5;scoutHead=shared`)',
      role: 'Reverse-LM scout sharing forward readout $W_{\\text{frontier}}, W_{\\text{present}}$',
      recs: v14DefRecs
    },
    {
      label: '**A4** vs v0.0014 default † *(paired readout isolation)*',
      role: 'Effect of separating scout readout head at `scoutScale=0.5`',
      recs: getFreshSeries('L12:arch=learns;dim=32;lr=0.016;scoutScale=0.5;scoutHead=separate', 1000, 'stride', seeds),
      refRecs: v14DefRecs,
      refNote: 'vs shared scout †'
    },
    {
      label: 'GRU `ternary` (`dim=32, lr=0.016`)',
      role: 'Parameter-matched 2-layer ternary GRU baseline',
      recs: getReusedSeries('L12:arch=gru;dim=32;lr=0.016', 1000, 'stride', seeds, '0x4048aacb')
    }
  ];

  console.log('### Table 1A: Tier 2 Partner Record Study — Held-Out Test Loss & Verdicts (1,000 Steps, `--stream=stride`, 5 Seeds `3, 7, 11, 13, 17`, `--tail 5 --final`)');
  console.log('');
  console.log('| Arm / Configuration | Partner Record Mechanism | Val Mean | Mean Tail Range | Test Mean | Test Seeds (`3 / 7 / 11 / 13 / 17`) | Paired $\\Delta\\text{Test}$ Mean $\\pm s_\\Delta$ | 95% Student-$t$ CI ($df=4$) | Max Tail Range | Tier 1 Verdict |');
  console.log('|---|---|---:|---:|---:|---|---:|---|---:|---|');

  for (const r of rows) {
    const recs = r.recs;
    if (!recs || recs.some(x => !x)) continue;
    const lbl = markLabelIfReused(r.label, recs, r.refRecs);
    const valMean = fmt(mean(recs.map(x => x.valLoss)));
    const tailRange = fmt(mean(recs.map(x => x.valTailRange)));
    const testMean = fmt(mean(recs.map(x => x.testLoss)));
    const testSeeds = recs.map(x => fmt(x.testLoss)).join(' / ');
    if (r.isBase) {
      console.log(`| ${lbl} | ${r.role} | ${valMean} | ${tailRange} | **${testMean}** | ${testSeeds} | — *(baseline B0)* | — | ${tailRange} | **Baseline** |`);
    } else {
      const ref = r.refRecs || b0Recs;
      const st = computePairedStats(ref, recs, 'testLoss');
      const ciStr = `[${fmtSigned(st.ciLo)}, ${fmtSigned(st.ciHi)}]`;
      const note = r.refNote ? ` *(${r.refNote})*` : '';
      console.log(`| ${lbl} | ${r.role} | ${valMean} | ${tailRange} | ${testMean} | ${testSeeds} | ${fmtSigned(st.dMean)} $\\pm$ ${fmt(st.dSD)}${note} | \`${ciStr}\` | ${fmt(st.maxTailRange)} | **${st.verdict}** |`);
    }
  }
  console.log('');
  console.log('*Note: Rows marked with `†` reuse bit-identical runs from `results.jsonl` with their original `codeHash` (`0x4048aacb` for v0.0014) and `reusedFrom` provenance field preserved.*');
  console.log('');
}

function printTier2DiagTable() {
  const seeds = [3, 7, 11, 13, 17];
  const b0Recs = getFreshSeries('L12:arch=learns;dim=32;lr=0.016', 1000, 'stride', seeds);
  const v14DefRecs = getReusedSeries('L12:arch=learns;dim=32;lr=0.016', 1000, 'stride', seeds, '0x4048aacb');
  const v14Seed3Fresh = getFreshRec('L12:arch=learns;dim=32;lr=0.016;scoutScale=0.5', 1000, 'stride', 3);

  const rows = [
    { label: '**B0**: `scoutScale=0;rbMode=causal` *(default)*', recs: b0Recs, isBase: true },
    { label: '**A1**: `recordB=off` (`scoutScale=0`)', recs: getFreshSeries('L12:arch=learns;dim=32;lr=0.016;recordB=off', 1000, 'stride', seeds) },
    { label: '**A2**: `rbMode=prevrev` (`scoutScale=0`)', recs: getFreshSeries('L12:arch=learns;dim=32;lr=0.016;rbMode=prevrev', 1000, 'stride', seeds) },
    { label: '**A3a**: `rbMode=lookback;lookback=16` (`scoutScale=0`)', recs: getFreshSeries('L12:arch=learns;dim=32;lr=0.016;rbMode=lookback;lookback=16', 1000, 'stride', seeds) },
    { label: '**A3b**: `rbMode=lookback;lookback=64` (`scoutScale=0`)', recs: getFreshSeries('L12:arch=learns;dim=32;lr=0.016;rbMode=lookback;lookback=64', 1000, 'stride', seeds) },
    { label: '**A4**: `scoutScale=0.5;scoutHead=separate`', recs: getFreshSeries('L12:arch=learns;dim=32;lr=0.016;scoutScale=0.5;scoutHead=separate', 1000, 'stride', seeds) },
    { label: 'v0.0014 default (`scoutScale=0.5;scoutHead=shared`)', recs: v14DefRecs, diagFallback: v14Seed3Fresh }
  ];

  console.log('### Table 1B: Tier 2 Partner Record Study — Validation Verdicts, Training Speed (`tokensPerSec`), and End-of-Training $\\mathbf{R}_B$ Diagnostics (1,000 Steps, 5 Seeds `3, 7, 11, 13, 17`)');
  console.log('');
  console.log('| Arm / Configuration | Params | Val Mean | Val Tail Mean | Mean Tail Range | Paired $\\Delta\\text{Val}$ Mean $\\pm s_\\Delta$ | 95% Student-$t$ CI ($df=4$) | Tier 1 Verdict (Val) | Mean `tokensPerSec` | End $\\|\\mathbf{R}_A\\|_2$ | End $\\|\\mathbf{R}_B\\|_2$ | End $\\cos(\\mathbf{R}_A, \\mathbf{R}_B)$ |');
  console.log('|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|');

  for (const r of rows) {
    const recs = r.recs;
    if (!recs || recs.some(x => !x)) continue;
    const lbl = markLabelIfReused(r.label, recs);
    const params = recs[0].paramCount.toLocaleString('en-US');
    const valMean = fmt(mean(recs.map(x => x.valLoss)));
    const tailMean = fmt(mean(recs.map(x => x.valTailMean)));
    const tailRange = fmt(mean(recs.map(x => x.valTailRange)));
    const tps = Math.round(mean(recs.map(x => x.tokensPerSec))).toLocaleString('en-US');

    let rANormStr, rBNormStr, cosStr;
    if (recs.every(x => typeof x.rBNorm === 'number')) {
      rANormStr = fmt(mean(recs.map(x => x.rANorm)), 4);
      rBNormStr = fmt(mean(recs.map(x => x.rBNorm)), 4);
      cosStr = fmtSigned(mean(recs.map(x => x.rArBCos)), 4);
    } else if (r.diagFallback) {
      rANormStr = `${fmt(r.diagFallback.rANorm, 4)} *(s3)*`;
      rBNormStr = `${fmt(r.diagFallback.rBNorm, 4)} *(s3)*`;
      cosStr = `${fmtSigned(r.diagFallback.rArBCos, 4)} *(s3)*`;
    } else {
      rANormStr = '—'; rBNormStr = '—'; cosStr = '—';
    }

    if (r.isBase) {
      console.log(`| ${lbl} | ${params} | **${valMean}** | ${tailMean} | ${tailRange} | — *(baseline B0)* | — | **Baseline** | ${tps} | ${rANormStr} | ${rBNormStr} | ${cosStr} |`);
    } else {
      const st = computePairedStats(b0Recs, recs, 'valLoss');
      const ciStr = `[${fmtSigned(st.ciLo)}, ${fmtSigned(st.ciHi)}]`;
      console.log(`| ${lbl} | ${params} | ${valMean} | ${tailMean} | ${tailRange} | ${fmtSigned(st.dMean)} $\\pm$ ${fmt(st.dSD)} | \`${ciStr}\` | **${st.verdict}** | ${tps} | ${rANormStr} | ${rBNormStr} | ${cosStr} |`);
    }
  }
  console.log('');
}

function printTier2TrajectoryTable() {
  const seeds = [3, 7, 11, 13, 17];
  const arms = [
    { label: '**B0**: `scoutScale=0;rbMode=causal`', cfg: 'L12:arch=learns;dim=32;lr=0.016' },
    { label: '**A1**: `recordB=off`', cfg: 'L12:arch=learns;dim=32;lr=0.016;recordB=off' },
    { label: '**A2**: `rbMode=prevrev`', cfg: 'L12:arch=learns;dim=32;lr=0.016;rbMode=prevrev' },
    { label: '**A3a**: `rbMode=lookback;lookback=16`', cfg: 'L12:arch=learns;dim=32;lr=0.016;rbMode=lookback;lookback=16' },
    { label: '**A3b**: `rbMode=lookback;lookback=64`', cfg: 'L12:arch=learns;dim=32;lr=0.016;rbMode=lookback;lookback=64' },
    { label: '**A4**: `scoutScale=0.5;scoutHead=separate`', cfg: 'L12:arch=learns;dim=32;lr=0.016;scoutScale=0.5;scoutHead=separate' },
    { label: 'v0.0014 default (`scoutScale=0.5;scoutHead=shared`, seed 3 repro)', cfg: 'L12:arch=learns;dim=32;lr=0.016;scoutScale=0.5', seedsOverride: [3] }
  ];

  const stepsList = [1, 250, 500, 750, 1000];
  console.log('### Table 1C: Partner Record $\\mathbf{R}_B$ Norm $\\|\\mathbf{R}_B\\|_2$ and Cosine $\\cos(\\mathbf{R}_A, \\mathbf{R}_B)$ Trajectory Over Training Time (5-Seed Means across Steps `1 .. 1000` and Final Validation)');
  console.log('');
  console.log('| Arm / Configuration | Step 1 $\\|\\mathbf{R}_B\\| / \\cos$ | Step 250 $\\|\\mathbf{R}_B\\| / \\cos$ | Step 500 $\\|\\mathbf{R}_B\\| / \\cos$ | Step 750 $\\|\\mathbf{R}_B\\| / \\cos$ | Step 1000 $\\|\\mathbf{R}_B\\| / \\cos$ | Final Val $\\|\\mathbf{R}_B\\| / \\cos$ |');
  console.log('|---|---|---|---|---|---|---|');

  for (const a of arms) {
    const sList = a.seedsOverride || seeds;
    const recs = getFreshSeries(a.cfg, 1000, 'stride', sList);
    if (recs.some(x => !x || !x.rBDiagCurve)) continue;
    const stepCells = stepsList.map(st => {
      const pts = recs.map(r => r.rBDiagCurve.find(p => p.step === st)).filter(Boolean);
      const nMean = mean(pts.map(p => p.rBNorm));
      const cMean = mean(pts.map(p => p.rArBCos));
      return `${fmt(nMean, 2)} / ${fmtSigned(cMean, 3)}`;
    });
    const valN = mean(recs.map(r => r.rBNorm));
    const valC = mean(recs.map(r => r.rArBCos));
    console.log(`| ${a.label} | ${stepCells.join(' | ')} | **${fmt(valN, 2)} / ${fmtSigned(valC, 3)}** |`);
  }
  console.log('');
}

function printTier2LRSweepTable() {
  console.log('### Table 1D: Validation Learning-Rate Verification for Partner-Record Modes (`seed=3`, 1,000 Steps, `--stream=stride`, `--tail 5`, Validation Only)');
  console.log('');
  console.log('| Partner-Record Mode | Candidate `lr` | Config String | Val Final | Val Tail Min | Val Tail Max | Val Tail Mean | Val Tail Range | Selected on Val? |');
  console.log('|---|---:|---|---:|---:|---:|---:|---:|---|');

  const groups = [
    {
      name: '**B0**: `scoutScale=0;rbMode=causal`',
      items: [
        { lr: '0.008', rec: getReusedRec('L12:arch=learns;dim=32;lr=0.008;scoutScale=0', 1000, 'stride', 3, '0x4048aacb') },
        { lr: '0.016', rec: getFreshRec('L12:arch=learns;dim=32;lr=0.016', 1000, 'stride', 3) },
        { lr: '0.024', rec: getReusedRec('L12:arch=learns;dim=32;lr=0.024;scoutScale=0', 1000, 'stride', 3, '0x4048aacb') }
      ]
    },
    {
      name: '**A3a**: `rbMode=lookback;lookback=16`',
      items: [
        { lr: '0.008', rec: getFreshRec('L12:arch=learns;dim=32;lr=0.008;rbMode=lookback;lookback=16', 1000, 'stride', 3) },
        { lr: '0.016', rec: getFreshRec('L12:arch=learns;dim=32;lr=0.016;rbMode=lookback;lookback=16', 1000, 'stride', 3) },
        { lr: '0.024', rec: getFreshRec('L12:arch=learns;dim=32;lr=0.024;rbMode=lookback;lookback=16', 1000, 'stride', 3) }
      ]
    },
    {
      name: '**A3b**: `rbMode=lookback;lookback=64`',
      items: [
        { lr: '0.008', rec: getFreshRec('L12:arch=learns;dim=32;lr=0.008;rbMode=lookback;lookback=64', 1000, 'stride', 3) },
        { lr: '0.016', rec: getFreshRec('L12:arch=learns;dim=32;lr=0.016;rbMode=lookback;lookback=64', 1000, 'stride', 3) },
        { lr: '0.024', rec: getFreshRec('L12:arch=learns;dim=32;lr=0.024;rbMode=lookback;lookback=64', 1000, 'stride', 3) }
      ]
    },
    {
      name: '**A4**: `scoutScale=0.5;scoutHead=separate`',
      items: [
        { lr: '0.008', rec: getFreshRec('L12:arch=learns;dim=32;lr=0.008;scoutScale=0.5;scoutHead=separate', 1000, 'stride', 3) },
        { lr: '0.016', rec: getFreshRec('L12:arch=learns;dim=32;lr=0.016;scoutScale=0.5;scoutHead=separate', 1000, 'stride', 3) },
        { lr: '0.024', rec: getFreshRec('L12:arch=learns;dim=32;lr=0.024;scoutScale=0.5;scoutHead=separate', 1000, 'stride', 3) }
      ]
    }
  ];

  for (const g of groups) {
    let bestLR = null, bestVal = Infinity;
    for (const it of g.items) {
      if (it.rec && it.rec.valLoss < bestVal) { bestVal = it.rec.valLoss; bestLR = it.lr; }
    }
    for (const it of g.items) {
      const r = it.rec;
      if (!r) continue;
      const win = it.lr === bestLR;
      const rowName = r.reusedFrom ? `${g.name} †` : g.name;
      const cfgDagger = r.reusedFrom ? ' †' : '';
      console.log(`| ${rowName} | \`${it.lr}\` | \`${r.config}\`${cfgDagger} | ${win ? '**' + fmt(r.valLoss) + '**' : fmt(r.valLoss)} | ${fmt(r.valTailMin)} | ${fmt(r.valTailMax)} | ${fmt(r.valTailMean)} | ${fmt(r.valTailRange)} | ${win ? '**Yes (Winner)**' : 'No'} |`);
    }
  }
  console.log('');
}

function printTier3Tables() {
  const seeds = [3, 7, 11];
  const b0FullRecs = getFreshSeries('L12:arch=learns;dim=32;lr=0.016', 7030, 'stride', seeds);
  const gruFullFreshRecs = getFreshSeries('L12:arch=gru;dim=32;lr=0.016', 7030, 'stride', seeds);

  const arms = [
    {
      label: 'LEARNS v0.0015 default (**B0**: `scoutScale=0, rbMode=causal`)',
      code: '`0x626fd26c` *(fresh v0.0015)*',
      recs: b0FullRecs,
      isBase: true
    },
    {
      label: 'GRU `ternary` (`dim=32, lr=0.016`)',
      code: '`0x626fd26c` *(fresh v0.0015)*',
      recs: gruFullFreshRecs
    },
    {
      label: 'LEARNS v0.0014 default (`scoutScale=0.5, rescale, inject`)',
      code: '`0x4048aacb` *(v0.0014)*',
      recs: getReusedSeries('L12:arch=learns;dim=32;lr=0.016', 7030, 'stride', seeds, '0x4048aacb')
    },
    {
      label: 'LEARNS `weights=float32;scoutScale=0` (`lr=0.016`)',
      code: '`0x4048aacb` *(v0.0014)*',
      recs: getReusedSeries('L12:arch=learns;dim=32;lr=0.016;weights=float32;scoutScale=0', 7030, 'stride', seeds, '0x4048aacb')
    },
    {
      label: 'LEARNS `weights=float32;scoutScale=0.5` (`lr=0.016`)',
      code: '`0x4048aacb` *(v0.0014)*',
      recs: getReusedSeries('L12:arch=learns;dim=32;lr=0.016;weights=float32', 7030, 'stride', seeds, '0x4048aacb')
    },
    {
      label: 'LEARNS v0.0013 legacy (`layernorm, no-inject, scoutScale=0.5`)',
      code: '`0xdf8ae499` *(v0.0013)*',
      recs: getReusedSeries('L12:arch=learns;dim=32;lr=0.016;fpInject=off;fpNorm=layernorm', 7030, 'stride', seeds, '0xdf8ae499')
    },
    {
      label: 'Transformer `ternary` (`dim=32, lr=0.016`)',
      code: '`0xdf8ae499` *(v0.0013)*',
      recs: getReusedSeries('L12:arch=transformer;dim=32;lr=0.016', 7030, 'stride', seeds, '0xdf8ae499')
    },
    {
      label: 'Transformer `float32` (`dim=32, lr=0.008`)',
      code: '`0xdf8ae499` *(v0.0013)*',
      recs: getReusedSeries('L12:arch=transformer;dim=32;lr=0.008;weights=float32', 7030, 'stride', seeds, '0xdf8ae499')
    },
    {
      label: 'GRU `float32` (`dim=32, lr=0.016`)',
      code: '`0x4048aacb` *(v0.0014)*',
      recs: getReusedSeries('L12:arch=gru;dim=32;lr=0.016;weights=float32', 7030, 'stride', seeds, '0x4048aacb')
    },
    {
      label: 'GRU `float32` (`dim=32, lr=0.008`)',
      code: '`0xdf8ae499` *(v0.0013)*',
      recs: getReusedSeries('L12:arch=gru;dim=32;lr=0.008;weights=float32', 7030, 'stride', seeds, '0xdf8ae499')
    }
  ];

  console.log('### Table 2A: Tier 3 Full-Book Training Pass (`7,030` Steps = 1.0 Epoch, `--stream=stride`, 3 Seeds `3, 7, 11`, `--tail 5 --final`)');
  console.log('');
  console.log('| Architecture / Configuration | `codeHash` | Val Mean | Mean Tail Range | Test Mean | Test Seeds (`3 / 7 / 11`) | Paired $\\Delta\\text{Test}$ vs B0 | 95% Student-$t$ CI ($df=2$) | Max Tail Range | Tier 1 Verdict |');
  console.log('|---|---|---:|---:|---:|---|---:|---|---:|---|');

  for (const a of arms) {
    const recs = a.recs;
    if (!recs || recs.some(x => !x)) continue;
    const lbl = markLabelIfReused(a.label, recs);
    const valMean = fmt(mean(recs.map(x => x.valLoss)));
    const tailRange = hasMultiTail(recs) ? fmt(mean(recs.map(x => x.valTailRange))) : '—';
    const testMean = fmt(mean(recs.map(x => x.testLoss)));
    const testSeeds = recs.map(x => fmt(x.testLoss)).join(' / ');
    if (a.isBase) {
      console.log(`| ${lbl} | ${a.code} | ${valMean} | ${tailRange} | **${testMean}** | ${testSeeds} | — *(baseline B0)* | — | ${tailRange} | **Baseline** |`);
    } else {
      const st = computePairedStats(b0FullRecs, recs, 'testLoss');
      const ciStr = `[${fmtSigned(st.ciLo)}, ${fmtSigned(st.ciHi)}]`;
      console.log(`| ${lbl} | ${a.code} | ${valMean} | ${tailRange} | ${testMean} | ${testSeeds} | ${fmtSigned(st.dMean)} $\\pm$ ${fmt(st.dSD)} | \`${ciStr}\` | ${fmt(st.maxTailRange)} | **${st.verdict}** |`);
    }
  }
  console.log('');
  console.log('*Note: The first two rows (`LEARNS v0.0015 default B0` and `GRU ternary`) are fresh v0.0015 runs (`codeHash: 0x626fd26c`) with full 5-checkpoint tail bands (`--tail 5 --final`). Context rows marked with `†` reuse prior-release runs with their original `codeHash` (`0x4048aacb` or `0xdf8ae499`) and `reusedFrom` field preserved.*');
  console.log('');

  const checkpoints = [1000, 2000, 3000, 4000, 5000, 6000, 7030];
  console.log('### Table 2B: Full-Pass Validation Learning Curves (3-Seed Mean Validation Loss in nats/byte across Checkpoints `1,000 .. 7,030`)');
  console.log('');
  console.log('| Architecture / Configuration | `codeHash` | Step 1,000 | Step 2,000 | Step 3,000 | Step 4,000 | Step 5,000 | Step 6,000 | Step 7,030 (Final) |');
  console.log('|---|---|---:|---:|---:|---:|---:|---:|---:|');

  for (const a of arms) {
    const recs = a.recs;
    if (!recs || recs.some(x => !x || !x.valCurve)) continue;
    const lbl = markLabelIfReused(a.label, recs);
    const cols = checkpoints.map(cp => {
      const vals = recs.map(r => {
        const pt = r.valCurve.find(p => p.step === cp);
        return pt ? pt.valLoss : NaN;
      });
      return fmt(mean(vals));
    });
    console.log(`| ${lbl} | ${a.code} | ${cols.slice(0, 6).join(' | ')} | **${cols[6]}** |`);
  }
  console.log('');
}

if (secArg === 'all' || secArg === 'tier2') printTier2MainTable();
if (secArg === 'all' || secArg === 'tier2_diag') printTier2DiagTable();
if (secArg === 'all' || secArg === 'tier2_traj') printTier2TrajectoryTable();
if (secArg === 'all' || secArg === 'tier2_lr') printTier2LRSweepTable();
if (secArg === 'all' || secArg === 'tier3') printTier3Tables();