#!/usr/bin/env node
// Checkpoint jitter: how much does full-split validation loss move between the last few steps of a run,
// when the cosine schedule is already at its floor (lrMin = 1e-5)? Mirrors tests/run.js's training loop exactly.
// Usage: node tests/jitter.js --arch learns --config "dim=32;lr=0.016" --seed 3 --steps 1000 --at 970,980,990,1000
'use strict';
const fs = require('fs'), path = require('path');
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i >= 0 ? process.argv[i + 1] : d; };
const L = require('./load.js')(path.join(__dirname, '..', 'index.html'), { f64: false, ternary: true });
const cfg = L.parseConfig(arg('config', ''), { arch: arg('arch', 'learns') });
cfg.arch = arg('arch', 'learns');
const seed = Number(arg('seed', 3)), steps = Number(arg('steps', 1000));
const at = new Set(arg('at', '').split(',').map(Number));
L.Tokenizer.useByteLevel();
const { train, val } = L.splitCorpus80_10_10(Int32Array.from(fs.readFileSync(path.join(__dirname, '..', 'corpus.txt'))));
(async () => {
  L.seedRNG(seed >>> 0);
  const model = L.createModel(cfg);
  Object.assign(L.App, { model, trainState: model.getEmptyState(), prevScoutRB: null, phiClip: cfg.phiClip !== 'off', clipMode: cfg.clipMode || 'global', fpNorm: cfg.fpNorm || 'layernorm' });
  const cs = cfg.chunk || 64, starts = L.buildStrideStarts(train.length, cs, steps, seed);
  let prev;
  const out = [];
  for (let s = 0; s < steps; s++) {
    const st = starts[s];
    cfg.contiguous = prev !== undefined && st === prev + cs;
    if (!cfg.contiguous) { L.App.trainState = model.getEmptyState(); L.App.prevScoutRB = null; }
    prev = st;
    await L.trainChunk(train.subarray(st, st + cs + 1), cfg);
    L.optimizerStep(model, s + 1, L.cosineLR(s + 1, cfg.lr, cfg.lrMin, cfg.warmup, steps), cfg);
    if (at.has(s + 1)) { const v = L.evalFullSplit(val, model, cfg, cs); out.push([s + 1, +v.loss.toFixed(6)]); }
  }
  console.log(JSON.stringify({ arch: cfg.arch, config: arg('config', ''), seed, steps, valAt: out }));
})();
