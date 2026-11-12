#!/usr/bin/env node
// Legacy bit-identity verification test for LEARNS v0.0016:
// 1. Proves that the explicit v0.0011/v0.0012 configuration:
//      dim=32;layers=2;chunk=64;lr=0.004;fpNorm=layernorm;fpInject=off;helixState=shared;scoutScale=0.5;ema=off
//    with --stream=head reproduces v0.0011 and v0.0012 bit-for-bit on corpus.txt across all 40 steps
//    and matches the 150-step v0.0012 baseline in results.jsonl.
// 2. Proves that the explicit v0.0014 default configuration:
//      dim=32;layers=2;chunk=64;lr=0.016;fpNorm=rescale;fpInject=on;scoutScale=0.5
//    with --stream=stride (seed 3, 1,000 steps, --tail 5 --final) reproduces its v0.0014 record bit-for-bit.
// Usage: node tests/repro-v0011.js
'use strict';
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const htmlPath = path.join(__dirname, '..', 'index.html');
const corpusPath = path.join(__dirname, '..', 'corpus.txt');
const L = require('./load.js')(htmlPath, { f64: false, ternary: true });

const LEGACY_CONFIG = 'dim=32;layers=2;chunk=64;lr=0.004;fpNorm=layernorm;fpInject=off;helixState=shared;scoutScale=0.5;ema=off';
const V0014_DEFAULT_CONFIG = 'dim=32;layers=2;chunk=64;lr=0.016;fpNorm=rescale;fpInject=on;scoutScale=0.5';

const EXPECTED_STEPS = {
  1:  { loss: '5.544955097855', thermo: '0.017997727164' },
  5:  { loss: '5.525352397877', thermo: '0.001938884091' },
  10: { loss: '5.492313755198', thermo: '0.000000000000' },
  25: { loss: '4.821015726850', thermo: '0.000417984064' },
  40: { loss: '3.509413177366', thermo: '0.000017065644' }
};

(async () => {
  L.Tokenizer.useByteLevel();
  const corpusBytes = fs.readFileSync(corpusPath);
  const { train } = L.splitCorpus80_10_10(Int32Array.from(corpusBytes));
  const cfg = L.parseConfig(LEGACY_CONFIG, { arch: 'learns' });

  L.seedRNG(3);
  const model = L.createModel(cfg);
  Object.assign(L.App, {
    model,
    trainState: model.getEmptyState(),
    prevScoutRB: null,
    phiClip: true,
    clipMode: 'global',
    fpNorm: 'layernorm'
  });

  let fails = 0;
  for (let s = 0; s < 40; s++) {
    const start = (s * 64) % Math.max(1, train.length - 66);
    const chunk = train.subarray(start, start + 65);
    cfg.contiguous = (s > 0);
    const lr = L.cosineLR(s + 1, cfg.lr, cfg.lrMin, cfg.warmup, 150);
    const r = await L.trainChunk(chunk, cfg);
    L.optimizerStep(model, s + 1, lr, cfg);
    const stepNum = s + 1;
    if (EXPECTED_STEPS[stepNum]) {
      const gotLoss = r.loss.toFixed(12);
      const gotThermo = r.thermoLoss.toFixed(12);
      const exp = EXPECTED_STEPS[stepNum];
      const ok = (gotLoss === exp.loss && gotThermo === exp.thermo);
      if (!ok) fails++;
      console.log(`step ${String(stepNum).padStart(2)}: loss=${gotLoss} (exp ${exp.loss}) thermo=${gotThermo} (exp ${exp.thermo}) ${ok ? 'ok' : 'FAIL'}`);
    }
  }

  // Verify 150-step v0.0012 run.js output under --stream=head
  const outLine = execFileSync(process.execPath, [
    path.join(__dirname, 'run.js'),
    '--arch', 'learns',
    '--config', LEGACY_CONFIG,
    '--seed', '3',
    '--steps', '150',
    '--stream', 'head',
    '--tail', '1',
    '--final'
  ], { encoding: 'utf8' }).trim();
  const rec = JSON.parse(outLine);
  const exp150 = {
    trainLossFirst: 5.5449551,
    trainLossLast: 3.32738447,
    valLoss: 3.137877,
    testLoss: 3.152259
  };
  for (const [k, v] of Object.entries(exp150)) {
    const ok = rec[k] === v;
    if (!ok) fails++;
    console.log(`run.js 150-step v0.0012 ${k.padEnd(16)}: got=${rec[k]} exp=${v} ${ok ? 'ok' : 'FAIL'}`);
  }

  // Verify 1,000-step v0.0014 default (scoutScale=0.5, seed 3, --stream=stride) bit-for-bit
  const out14 = execFileSync(process.execPath, [
    path.join(__dirname, 'run.js'),
    '--arch', 'learns',
    '--config', V0014_DEFAULT_CONFIG,
    '--seed', '3',
    '--steps', '1000',
    '--stream', 'stride',
    '--tail', '5',
    '--final'
  ], { encoding: 'utf8' }).trim();
  const rec14 = JSON.parse(out14);
  const exp1000 = {
    trainLossFirst: 5.54189737,
    trainLossLast: 2.15520647,
    trainLoss: 2.299218,
    valLoss: 2.321951,
    valTailMin: 2.319591,
    valTailMax: 2.339794,
    valTailMean: 2.326029,
    testLoss: 2.365709
  };
  for (const [k, v] of Object.entries(exp1000)) {
    const ok = rec14[k] === v;
    if (!ok) fails++;
    console.log(`run.js 1000-step v0.0014 ${k.padEnd(15)}: got=${rec14[k]} exp=${v} ${ok ? 'ok' : 'FAIL'}`);
  }

  if (fails > 0) {
    console.error(`FAILED: ${fails} mismatch(es) in legacy v0.0011/v0.0012/v0.0014 reproduction`);
    process.exit(1);
  }
  console.log(`ALL OK (legacy v0.0011/v0.0012 --stream=head and v0.0014 1,000-step scoutScale=0.5 reproduce bit-for-bit)`);
})();