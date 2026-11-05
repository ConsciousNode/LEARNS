#!/usr/bin/env node
// Legacy bit-identity verification test for LEARNS v0.0014:
// Proves that the explicit legacy configuration:
//   dim=32;layers=2;chunk=64;lr=0.004;fpNorm=layernorm;fpInject=off;helixState=shared;scoutScale=0.5;ema=off
// with --stream=head reproduces v0.0011 and v0.0012 bit-for-bit on corpus.txt across all 40 steps
// and matches the 150-step v0.0012 baseline in results.jsonl.
// Usage: node tests/repro-v0011.js
'use strict';
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const htmlPath = path.join(__dirname, '..', 'index.html');
const corpusPath = path.join(__dirname, '..', 'corpus.txt');
const L = require('./load.js')(htmlPath, { f64: false, ternary: true });

const LEGACY_CONFIG = 'dim=32;layers=2;chunk=64;lr=0.004;fpNorm=layernorm;fpInject=off;helixState=shared;scoutScale=0.5;ema=off';

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

  // Also verify 150-step run.js output under --stream=head
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
    console.log(`run.js 150-step ${k.padEnd(16)}: got=${rec[k]} exp=${v} ${ok ? 'ok' : 'FAIL'}`);
  }

  if (fails > 0) {
    console.error(`FAILED: ${fails} mismatch(es) in legacy v0.0011/v0.0012 reproduction`);
    process.exit(1);
  }
  console.log(`ALL OK (legacy config "${LEGACY_CONFIG}" with --stream=head reproduces v0.0011/v0.0012 bit-for-bit)`);
})();