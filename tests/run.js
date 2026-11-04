#!/usr/bin/env node
// Headless experiment runner for LEARNS v0.0012: The Honest Lab
// Loads the marked slice /*@@LEARNS_BEGIN*/ ... /*@@LEARNS_END*/ from index.html,
// verifies/computes CODE_HASH and CORPUS_HASH, runs deterministic 80/10/10 split,
// trains for --steps, evaluates causally over the full validation split (and test split only with --final),
// and prints one JSON line.
'use strict';
const fs = require('fs');
const path = require('path');

function parseArgs(argv) {
  const out = {
    corpus: 'corpus.txt',
    arch: null,
    config: '',
    seed: 3,
    steps: 400,
    final: false,
    curveEvery: 0,
    file: null
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === '--final') { out.final = true; continue; }
    if (a === '--corpus') { out.corpus = next(); continue; }
    if (a.startsWith('--corpus=')) { out.corpus = a.slice(9); continue; }
    if (a === '--arch') { out.arch = next(); continue; }
    if (a.startsWith('--arch=')) { out.arch = a.slice(7); continue; }
    if (a === '--config') { out.config = next(); continue; }
    if (a.startsWith('--config=')) { out.config = a.slice(9); continue; }
    if (a === '--seed') { out.seed = Number(next()); continue; }
    if (a.startsWith('--seed=')) { out.seed = Number(a.slice(7)); continue; }
    if (a === '--steps') { out.steps = Number(next()); continue; }
    if (a.startsWith('--steps=')) { out.steps = Number(a.slice(8)); continue; }
    if (a === '--curve') { out.curveEvery = Number(next()); continue; }
    if (a.startsWith('--curve=')) { out.curveEvery = Number(a.slice(8)); continue; }
    if (a === '--file') { out.file = next(); continue; }
    if (a.startsWith('--file=')) { out.file = a.slice(7); continue; }
  }
  return out;
}

const cli = parseArgs(process.argv.slice(2));
const htmlPath = cli.file || (fs.existsSync(path.join(__dirname, '..', 'index.html')) ? path.join(__dirname, '..', 'index.html') : path.resolve('index.html'));
const L = require('./load.js')(htmlPath, { f64: false, ternary: true });

const corpusPath = fs.existsSync(cli.corpus) ? cli.corpus : path.join(__dirname, '..', cli.corpus);
const corpusBytes = fs.readFileSync(corpusPath);
const CORPUS_HASH = L.fnv1aBytes(corpusBytes);
const CODE_HASH = L.CODE_HASH;

const baseCfg = {};
if (cli.arch) baseCfg.arch = cli.arch;
const cfg = L.parseConfig(cli.config, baseCfg);
if (cli.arch) {
  cfg.arch = cli.arch.toLowerCase();
  if (cfg.arch === 'unigram') cfg.ngramOrder = 1;
  else if (cfg.arch === 'bigram') cfg.ngramOrder = 2;
  else if (cfg.arch === 'trigram') cfg.ngramOrder = 3;
  else if (cfg.arch === '5gram' || cfg.arch === 'fivegram') cfg.ngramOrder = 5;
}

L.Tokenizer.useByteLevel();
const allTokens = Int32Array.from(corpusBytes);
const { train, val, test } = L.splitCorpus80_10_10(allTokens);

(async () => {
  const wallT0 = Date.now();
  L.seedRNG(cli.seed >>> 0);
  const model = L.createModel(cfg);
  L.App.model = model;
  L.App.trainState = model.getEmptyState();
  L.App.prevScoutRB = null;
  L.App.phiClip = cfg.phiClip !== 'off';
  L.App.clipMode = cfg.clipMode || 'global';

  const steps = (model.arch === 'ngram') ? 0 : cli.steps;
  const chunkSize = cfg.chunk || 64;
  let prevStart = undefined;
  let recentTrainLosses = [];
  let firstTrainLoss = null, lastTrainLoss = null;
  let spikeCount = 0, maxBetaGrad = 0;
  let fpResidualSum = 0, fpItersSum = 0, fpAgreeSum = 0, diagCount = 0;
  const valCurve = [];

  let curveMs = 0;
  const trnT0 = Date.now();
  if (model.arch === 'ngram') {
    model.fit(train);
  } else {
    for (let step = 0; step < steps; step++) {
      const start = (step * chunkSize) % Math.max(1, train.length - chunkSize - 2);
      const chunk = train.subarray(start, start + chunkSize + 1);
      cfg.contiguous = (prevStart !== undefined && start === prevStart + chunkSize);
      prevStart = start;
      const lr = L.cosineLR(step + 1, cfg.lr, cfg.lrMin, cfg.warmup, Math.max(1, steps));
      const { loss, thermoLoss, diag } = await L.trainChunk(chunk, cfg);
      L.optimizerStep(model, step + 1, lr, cfg);

      if (step === 0) firstTrainLoss = Number(loss.toFixed(8));
      lastTrainLoss = Number(loss.toFixed(8));
      recentTrainLosses.push(loss);
      if (recentTrainLosses.length > 20) recentTrainLosses.shift();

      if (model.lastGradStats) {
        if (model.lastGradStats.spiked) spikeCount++;
        if (model.lastGradStats.maxBetaGrad > maxBetaGrad) maxBetaGrad = model.lastGradStats.maxBetaGrad;
      }
      if (diag) {
        fpResidualSum += (diag.fpResidualAvg || 0);
        fpItersSum += (diag.fpItersAvg || 0);
        fpAgreeSum += (diag.fpAgreeAvg || 0);
        diagCount++;
      }

      if (cli.curveEvery > 0 && ((step + 1) % cli.curveEvery === 0 || step + 1 === steps)) {
        const ct0 = Date.now();
        const vc = L.evalFullSplit(val, model, cfg, chunkSize);
        curveMs += (Date.now() - ct0);
        valCurve.push({ step: step + 1, valLoss: Number(vc.loss.toFixed(6)), valBpb: Number(vc.bpb.toFixed(6)) });
      }
    }
  }
  const trainSec = Math.max(1e-6, (Date.now() - trnT0 - curveMs) / 1000);
  const valRes = L.evalFullSplit(val, model, cfg, chunkSize);
  const testRes = cli.final ? L.evalFullSplit(test, model, cfg, chunkSize) : null;
  const wallSec = (Date.now() - wallT0) / 1000;

  const trainLossMean = recentTrainLosses.length > 0
    ? recentTrainLosses.reduce((a, b) => a + b, 0) / recentTrainLosses.length
    : null;
  const align = model.getAlignmentDiag ? model.getAlignmentDiag() : null;
  const tokPerSec = steps > 0
    ? Math.round((steps * chunkSize) / trainSec)
    : Math.round((val.length - 1) / Math.max(1e-6, wallSec));

  const record = {
    codeHash: CODE_HASH,
    corpusHash: CORPUS_HASH,
    arch: cfg.arch,
    config: L.serializeConfig(cfg, true),
    seed: cli.seed,
    steps,
    paramCount: model.getParamCount(cfg),
    trainLoss: trainLossMean !== null ? Number(trainLossMean.toFixed(6)) : null,
    trainLossFirst: firstTrainLoss,
    trainLossLast: lastTrainLoss,
    valLoss: Number(valRes.loss.toFixed(6)),
    valBpb: Number(valRes.bpb.toFixed(6)),
    testLoss: testRes ? Number(testRes.loss.toFixed(6)) : null,
    testBpb: testRes ? Number(testRes.bpb.toFixed(6)) : null,
    bitsPerByte: Number((testRes ? testRes.bpb : valRes.bpb).toFixed(6)),
    wallClockSec: Number(wallSec.toFixed(2)),
    trainSec: Number(trainSec.toFixed(2)),
    tokensPerSec: tokPerSec,
    fpResidual: Number((valRes.fpResidual || (diagCount ? fpResidualSum / diagCount : 0)).toFixed(6)),
    fpItersAvg: Number((valRes.fpIters || (diagCount ? fpItersSum / diagCount : 0)).toFixed(3)),
    fpAgree: Number((diagCount ? fpAgreeSum / diagCount : 0).toFixed(6)),
    skewCos: align ? Number((align.skewCos || 0).toFixed(6)) : 0,
    rA_bias: align ? Number((align.rA_bias || 0).toFixed(6)) : 0,
    rB_bias: align ? Number((align.rB_bias || 0).toFixed(6)) : 0,
    divergence: align ? Number((align.divergence || 0).toFixed(6)) : 0,
    spikeCount,
    spikeFreq: steps > 0 ? Number((spikeCount / steps).toFixed(4)) : 0,
    maxBetaGrad: Number(maxBetaGrad.toExponential(3))
  };
  if (valCurve.length > 0) record.valCurve = valCurve;

  console.log(JSON.stringify(record));
})();