#!/usr/bin/env node
// Headless experiment runner for LEARNS v0.0014: "Signal and Noise"
// Loads the marked slice /*@@LEARNS_BEGIN*/ ... /*@@LEARNS_END*/ from index.html,
// verifies/computes CODE_HASH and CORPUS_HASH, runs deterministic 80/10/10 split,
// supports --stream=stride|block|head (default stride), --ngramData=all|seen (default all),
// and --tail K (default 5: scores full validation split at the final step and K-1 earlier checkpoints 10 steps apart),
// resets recurrent state across non-contiguous chunks identically for every architecture,
// trains for --steps, evaluates test split once at the final step only with --final,
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
    stepsExplicit: false,
    stream: 'stride',
    ngramData: 'all',
    tail: 5,
    final: false,
    curveEvery: 0,
    curveBytes: 0,
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
    if (a === '--steps') { out.steps = Number(next()); out.stepsExplicit = true; continue; }
    if (a.startsWith('--steps=')) { out.steps = Number(a.slice(8)); out.stepsExplicit = true; continue; }
    if (a === '--stream') { out.stream = String(next()).toLowerCase(); continue; }
    if (a.startsWith('--stream=')) { out.stream = a.slice(9).toLowerCase(); continue; }
    if (a === '--ngramData') { out.ngramData = String(next()).toLowerCase(); continue; }
    if (a.startsWith('--ngramData=')) { out.ngramData = a.slice(12).toLowerCase(); continue; }
    if (a === '--tail') { out.tail = Number(next()); continue; }
    if (a.startsWith('--tail=')) { out.tail = Number(a.slice(7)); continue; }
    if (a === '--curve') { out.curveEvery = Number(next()); continue; }
    if (a.startsWith('--curve=')) { out.curveEvery = Number(a.slice(8)); continue; }
    if (a === '--curveBytes') { out.curveBytes = Number(next()); continue; }
    if (a.startsWith('--curveBytes=')) { out.curveBytes = Number(a.slice(13)); continue; }
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

function buildStreamStarts(streamMode, trainLen, chunkSize, steps, seed) {
  if (streamMode === 'stride') return L.buildStrideStarts(trainLen, chunkSize, steps, seed);
  if (streamMode === 'block') return L.buildBlockStarts(trainLen, chunkSize, steps, seed);
  return Int32Array.from({ length: steps }, (_, s) => (s * chunkSize) % Math.max(1, trainLen - chunkSize - 2));
}

(async () => {
  const wallT0 = Date.now();
  L.seedRNG(cli.seed >>> 0);
  const model = L.createModel(cfg);
  L.App.model = model;
  L.App.trainState = model.getEmptyState();
  L.App.prevScoutRB = null;
  L.App.phiClip = cfg.phiClip !== 'off';
  L.App.clipMode = cfg.clipMode || 'global';
  L.App.fpNorm = cfg.fpNorm || 'rescale';

  const chunkSize = cfg.chunk || 64;
  const fullEpochSteps = Math.floor((train.length - 1) / chunkSize);
  const steps = (model.arch === 'ngram')
    ? (cli.ngramData === 'seen' ? (cli.stepsExplicit ? cli.steps : fullEpochSteps) : 0)
    : cli.steps;

  const tailK = Math.max(1, Number.isFinite(cli.tail) ? Math.floor(cli.tail) : 5);
  const earlyTailSteps = new Set();
  if (model.arch !== 'ngram' && tailK > 1 && steps > 0) {
    for (let k = tailK - 1; k >= 1; k--) {
      const stp = steps - 10 * k;
      if (stp >= 1) earlyTailSteps.add(stp);
    }
  }

  let prevStart = undefined;
  let recentTrainLosses = [];
  let firstTrainLoss = null, lastTrainLoss = null;
  let spikeCount = 0, maxBetaGrad = 0;
  let fpResidualSum = 0, fpItersSum = 0, fpAgreeSum = 0, diagCount = 0;
  const valCurve = [];
  const valTail = [];

  let evalMs = 0, evalCpuUs = 0;
  const trnT0 = Date.now();
  const trnCpu0 = process.cpuUsage();

  if (model.arch === 'ngram') {
    if (cli.ngramData === 'seen') {
      const starts = buildStreamStarts(cli.stream, train.length, chunkSize, steps, cli.seed);
      const segments = L.buildSeenSegments(train, starts, chunkSize);
      model.fitSegments(segments);
    } else {
      model.fit(train);
    }
  } else {
    const starts = buildStreamStarts(cli.stream, train.length, chunkSize, steps, cli.seed);

    for (let step = 0; step < steps; step++) {
      const start = starts[step];
      const chunk = train.subarray(start, start + chunkSize + 1);
      cfg.contiguous = (prevStart !== undefined && start === prevStart + chunkSize);
      if (!cfg.contiguous) {
        L.App.trainState = model.getEmptyState();
        L.App.prevScoutRB = null;
      }
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

      const curStep = step + 1;
      const needEarlyTail = earlyTailSteps.has(curStep);
      const needCurve = cli.curveEvery > 0 && curStep % cli.curveEvery === 0 && curStep < steps;
      if (needEarlyTail || needCurve) {
        const ct0 = Date.now();
        const cCpu0 = process.cpuUsage();
        if (needEarlyTail) {
          const vt = L.evalFullSplit(val, model, cfg, chunkSize);
          valTail.push([curStep, Number(vt.loss.toFixed(6))]);
          if (needCurve && cli.curveBytes <= 0) {
            valCurve.push({ step: curStep, valLoss: Number(vt.loss.toFixed(6)), valBpb: Number(vt.bpb.toFixed(6)) });
          } else if (needCurve) {
            const valTarget = val.subarray(0, Math.min(val.length, cli.curveBytes + 1));
            const vc = L.evalFullSplit(valTarget, model, cfg, chunkSize);
            valCurve.push({ step: curStep, valLoss: Number(vc.loss.toFixed(6)), valBpb: Number(vc.bpb.toFixed(6)) });
          }
        } else if (needCurve) {
          const valTarget = cli.curveBytes > 0 ? val.subarray(0, Math.min(val.length, cli.curveBytes + 1)) : val;
          const vc = L.evalFullSplit(valTarget, model, cfg, chunkSize);
          valCurve.push({ step: curStep, valLoss: Number(vc.loss.toFixed(6)), valBpb: Number(vc.bpb.toFixed(6)) });
        }
        const cCpu = process.cpuUsage(cCpu0);
        evalMs += (Date.now() - ct0);
        evalCpuUs += (cCpu.user + cCpu.system);
      }
    }
  }

  const trnCpu = process.cpuUsage(trnCpu0);
  const trainCpuSec = Math.max(1e-6, (trnCpu.user + trnCpu.system - evalCpuUs) / 1e6);
  const trainSec = Math.max(1e-6, (Date.now() - trnT0 - evalMs) / 1000);

  const valRes = L.evalFullSplit(val, model, cfg, chunkSize);
  const valLossFinal = Number(valRes.loss.toFixed(6));
  valTail.push([steps, valLossFinal]);
  if (model.arch !== 'ngram' && cli.curveEvery > 0 && steps > 0) {
    valCurve.push({ step: steps, valLoss: valLossFinal, valBpb: Number(valRes.bpb.toFixed(6)) });
  }
  const testRes = cli.final ? L.evalFullSplit(test, model, cfg, chunkSize) : null;
  const wallSec = (Date.now() - wallT0) / 1000;

  const tailVals = valTail.map(x => x[1]);
  const valTailMin = Number(Math.min(...tailVals).toFixed(6));
  const valTailMax = Number(Math.max(...tailVals).toFixed(6));
  const valTailMean = Number((tailVals.reduce((a, b) => a + b, 0) / tailVals.length).toFixed(6));
  const valTailRange = Number((valTailMax - valTailMin).toFixed(6));

  const trainLossMean = recentTrainLosses.length > 0
    ? recentTrainLosses.reduce((a, b) => a + b, 0) / recentTrainLosses.length
    : null;
  const align = model.getAlignmentDiag ? model.getAlignmentDiag() : null;
  const tokPerSec = (model.arch !== 'ngram' && steps > 0)
    ? Math.round((steps * chunkSize) / trainSec)
    : Math.round((val.length - 1) / Math.max(1e-6, wallSec));
  const cpuTokPerSec = (model.arch !== 'ngram' && steps > 0)
    ? Math.round((steps * chunkSize) / trainCpuSec)
    : tokPerSec;

  const record = {
    codeHash: CODE_HASH,
    corpusHash: CORPUS_HASH,
    arch: cfg.arch,
    config: L.serializeConfig(cfg, true),
    stream: cli.stream,
    ngramData: model.arch === 'ngram' ? cli.ngramData : undefined,
    seed: cli.seed,
    steps,
    paramCount: model.getParamCount(cfg),
    trainLoss: trainLossMean !== null ? Number(trainLossMean.toFixed(6)) : null,
    trainLossFirst: firstTrainLoss,
    trainLossLast: lastTrainLoss,
    valLoss: valLossFinal,
    valBpb: Number(valRes.bpb.toFixed(6)),
    valTail,
    valTailMin,
    valTailMax,
    valTailMean,
    valTailRange,
    testLoss: testRes ? Number(testRes.loss.toFixed(6)) : null,
    testBpb: testRes ? Number(testRes.bpb.toFixed(6)) : null,
    bitsPerByte: Number((testRes ? testRes.bpb : valRes.bpb).toFixed(6)),
    wallClockSec: Number(wallSec.toFixed(2)),
    trainSec: Number(trainSec.toFixed(2)),
    trainCpuSec: Number(trainCpuSec.toFixed(2)),
    tokensPerSec: tokPerSec,
    cpuTokensPerSec: cpuTokPerSec,
    fpResidual: Number((valRes.fpResidual || (diagCount ? fpResidualSum / diagCount : 0)).toFixed(6)),
    fpItersAvg: Number((valRes.fpIters || (diagCount ? fpItersSum / diagCount : 0)).toFixed(3)),
    fpAgree: Number((diagCount ? fpAgreeSum / diagCount : 0).toFixed(6)),
    skewCos: align ? Number((align.skewCos || 0).toFixed(6)) : 0,
    rA_bias: align ? Number((align.rA_bias || 0).toFixed(6)) : 0,
    rB_bias: align ? Number((align.rB_bias || 0).toFixed(6)) : 0,
    divergence: align ? Number((align.divergence || 0).toFixed(6)) : 0,
    spikeCount,
    spikeFreq: (model.arch !== 'ngram' && steps > 0) ? Number((spikeCount / steps).toFixed(4)) : 0,
    maxBetaGrad: Number(maxBetaGrad.toExponential(3))
  };
  if (valCurve.length > 0) record.valCurve = valCurve;

  console.log(JSON.stringify(record));
})();