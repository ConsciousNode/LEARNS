#!/usr/bin/env node
// Kehai's audit of v0.0016 (2026-11-11): paired verdicts for the comparisons the report states in prose but
// tables.js doesn't print, plus the Tier 3 ablations the delivery didn't finish (run by the audit,
// results-audit-v0016.jsonl). Uses tables.js's own pairedStats (t, df = 4; noise floor 0.015), unmodified.
'use strict';
const fs = require('fs'), path = require('path');
const src = fs.readFileSync(path.join(__dirname, 'tables.js'), 'utf8');
const pairedStats = new Function(src.slice(src.indexOf('function mean'), src.indexOf('return { n, meanDiff')) +
  'return { n, meanDiff: m, sdDiff: s, ciLo, ciHi, wins, verdict };\n}\nreturn pairedStats;')();
const read = f => fs.readFileSync(path.join(__dirname, '..', f), 'utf8').trim().split('\n').map(JSON.parse);
const all = read('results.jsonl').concat(read('results-audit-v0016.jsonl')).filter(r => r.tag);
for (const r of all) if (r.tag === 'v16_audit_t3_helixRet_fixed') r.tag = 'v16_t3_helixRet_fixed';
const SEEDS = [3, 7, 11, 13, 17];
const get = (tag, f) => SEEDS.map(s => { const r = all.find(x => x.tag === tag && x.seed === s); return r ? f(r) : NaN; });
const fmt = st => `${st.meanDiff >= 0 ? '+' : ''}${st.meanDiff.toFixed(4)} [${st.ciLo >= 0 ? '+' : ''}${st.ciLo.toFixed(4)}, ${st.ciHi >= 0 ? '+' : ''}${st.ciHi.toFixed(4)}] ${st.wins}/5 **${st.verdict}**`;
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const U = r => r.unseenTestLoss, C = r => r.crewTestLoss, R = k => r => r.unseenRoleLoss[k];
console.log('### Prose claims, recomputed\n\n| Comparison | Unseen Δ (95% CI, wins, verdict) | Crew Δ |\n| :-- | :-- | :-- |');
for (const [a, b, l] of [['v16_t2_learns_float32', 'v16_t2_transformer_float32', 'LEARNS f32 vs Transformer f32'],
  ['v16_t2_learns_ternary', 'v16_t2_transformer_ternary', 'LEARNS ternary vs Transformer ternary'],
  ['v16_t2_learns_ternary', 'v16_t2_gru_ternary', 'LEARNS ternary vs GRU ternary'],
  ['v16_t3_helixRot_none', 'v16_t2_transformer_ternary', 'helixRotation=none vs Transformer ternary']])
  console.log(`| ${l} | ${fmt(pairedStats(get(a, U), get(b, U)))} | ${fmt(pairedStats(get(a, C), get(b, C)))} |`);
console.log('\n### Tier 3, all arms, vs B0 ternary (lr 0.016)\n\n| Ablation | Unseen mean | Unseen Δ | Crew Δ | step Δ | pitch Δ | other Δ |\n| :-- | :-: | :-- | :-- | :-- | :-- | :-- |');
for (const [t, l] of [['v16_t3_helix_off', '`helix=off`'], ['v16_t3_helixRot_none', '`helixRotation=none`'],
  ['v16_t3_helixRet_fixed', '`helixRetention=fixed` (seed 3 Argon, 7–17 audit)'], ['v16_audit_t3_helixRet_none', '`helixRetention=none` (audit)'],
  ['v16_audit_t3_frontierPresent_off', '`frontierPresent=off` (audit)'], ['v16_audit_t3_recordB_off', '`recordB=off` (audit)']]) {
  const B = 'v16_t2_learns_ternary';
  console.log(`| ${l} | ${mean(get(t, U)).toFixed(4)} | ${fmt(pairedStats(get(t, U), get(B, U)))} | ${fmt(pairedStats(get(t, C), get(B, C)))} | ` +
    ['step', 'pitch', 'other'].map(k => fmt(pairedStats(get(t, R(k)), get(B, R(k))))).join(' | ') + ' |');
}
