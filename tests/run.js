#!/usr/bin/env node
// Headless experiment runner for LEARNS v0.0016: "Rhythm"
// Loads the marked slice /*@@LEARNS_BEGIN*/ ... /*@@LEARNS_END*/ from index.html,
// verifies/computes CODE_HASH and CORPUS_HASH, runs deterministic 80/10/10 prose split
// or procedural .loom music benchmark (--task=loom),
// supports --stream=stride|block|head (default stride), --ngramData=all|seen (default all),
// and --tail K (default 5: scores full validation split at the final step and K-1 earlier checkpoints 10 steps apart),
// resets recurrent state across non-contiguous chunks identically for every architecture,
// trains for --steps, evaluates test split(s) once at the final step only with --final,
// computes byte-role loss breakdown (step / pitch / other) as well as per-channel (26 unseen genres)
// and per-score (8 crew scores) losses on .loom, and prints one JSON line.
'use strict';
const fs = require('fs');
const path = require('path');

function parseArgs(argv) {
  const out = {
    task: 'prose',
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
    if (a === '--task') { out.task = String(next()).toLowerCase(); continue; }
    if (a.startsWith('--task=')) { out.task = a.slice(7).toLowerCase(); continue; }
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
  if (out.corpus.includes('loom-train')) out.task = 'loom';
  return out;
}

const cli = parseArgs(process.argv.slice(2));
const rootDir = path.join(__dirname, '..');
const htmlPath = cli.file || (fs.existsSync(path.join(rootDir, 'index.html')) ? path.join(rootDir, 'index.html') : path.resolve('index.html'));
const L = require('./load.js')(htmlPath, { f64: false, ternary: true });
const CODE_HASH = L.CODE_HASH;

const isLoomTask = cli.task === 'loom';
const loomCorpusMod = isLoomTask ? require('./loom-corpus.js') : null;

let CORPUS_HASH;
let train, val, test;
let valRoles = null;
let unseenChannels = null;
let crewScores = null;

L.Tokenizer.useByteLevel();

if (isLoomTask) {
  const trainPath = path.join(rootDir, 'loom-train.txt');
  const valPath = path.join(rootDir, 'loom-val.txt');
  const unseenJsonPath = path.join(rootDir, 'loom-test-unseen.json');
  const crewJsonPath = path.join(rootDir, 'loom-test-crew.json');
  const trainBytes = fs.readFileSync(trainPath);
  const valBytes = fs.readFileSync(valPath);
  CORPUS_HASH = L.fnv1aBytes(trainBytes);
  train = Int32Array.from(trainBytes);
  val = Int32Array.from(valBytes);
  valRoles = loomCorpusMod.classifyLoomByteRoles(valBytes);
  unseenChannels = JSON.parse(fs.readFileSync(unseenJsonPath, 'utf8')).map(ch => {
    const buf = Buffer.from(ch.text, 'utf8');
    return {
      id: ch.id,
      name: ch.name,
      cluster: ch.cluster,
      bytes: buf.length,
      tokens: Int32Array.from(buf),
      roles: loomCorpusMod.classifyLoomByteRoles(buf)
    };
  });
  crewScores = JSON.parse(fs.readFileSync(crewJsonPath, 'utf8')).map(sc => {
    const buf = Buffer.from(sc.text, 'utf8');
    return {
      id: sc.id,
      author: sc.author,
      bytes: buf.length,
      tokens: Int32Array.from(buf),
      roles: loomCorpusMod.classifyLoomByteRoles(buf)
    };
  });
} else {
  const corpusPath = fs.existsSync(cli.corpus) ? cli.corpus : path.join(rootDir, cli.corpus);
  const corpusBytes = fs.readFileSync(corpusPath);
  CORPUS_HASH = L.fnv1aBytes(corpusBytes);
  const allTokens = Int32Array.from(corpusBytes);
  const sp = L.splitCorpus80_10_10(allTokens);
  train = sp.train;
  val = sp.val;
  test = sp.test;
}

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

function buildStreamStarts(streamMode, trainLen, chunkSize, steps, seed) {
  if (streamMode === 'stride') return L.buildStrideStarts(trainLen, chunkSize, steps, seed);
  if (streamMode === 'block') return L.buildBlockStarts(trainLen, chunkSize, steps, seed);
  return Int32Array.from({ length: steps }, (_, s) => (s * chunkSize) % Math.max(1, trainLen - chunkSize - 2));
}

/**
 * Evaluates a token sequence `splitTokens` with optional `roleMask` (0=other, 1=step, 2=pitch).
 * Uses the exact same forward/chunk logic and numerical stability as L.evalFullSplit, while also
 * accumulating per-role NLL and counts in a single pass.
 */
function evalSplitWithRoles(splitTokens, roleMask, model, cfg = {}, chunkSize = 64) {
  const M = splitTokens.length;
  if (M < 2) {
    return {
      loss: 0, bpb: 0, nll: 0, count: 0,
      roleNLL: [0, 0, 0], roleCount: [0, 0, 0],
      roleLoss: { other: 0, step: 0, pitch: 0 },
      fpResidual: 0, fpIters: 0, rANorm: 0, rBNorm: 0, rArBCos: 0
    };
  }

  const roleNLL = [0, 0, 0];
  const roleCount = [0, 0, 0];

  if (model.arch === 'ngram') {
    let nll = 0;
    const hist = [];
    for (let i = 0; i < M - 1; i++) {
      hist.push(splitTokens[i]);
      if (hist.length > model.order - 1 && model.order > 1) hist.shift();
      const target = splitTokens[i + 1];
      const p = model.prob(target, model.order > 1 ? hist : []);
      const stepLoss = -Math.log(p + 1e-12);
      nll += stepLoss;
      const r = roleMask ? roleMask[i + 1] : 0;
      roleNLL[r] += stepLoss;
      roleCount[r]++;
    }
    const count = M - 1;
    const loss = nll / count;
    const bpb = (nll / Math.LN2) / count;
    return {
      loss, bpb, nll, count,
      roleNLL, roleCount,
      roleLoss: {
        other: roleCount[0] > 0 ? roleNLL[0] / roleCount[0] : null,
        step:  roleCount[1] > 0 ? roleNLL[1] / roleCount[1] : null,
        pitch: roleCount[2] > 0 ? roleNLL[2] / roleCount[2] : null
      },
      fpResidual: 0, fpIters: 0, rANorm: 0, rBNorm: 0, rArBCos: 0
    };
  }

  const evalCfg = Object.assign({}, cfg, { evalOnly: true });
  let state = model.getEmptyState();
  let prevScoutRB = null;
  const rbMode = cfg.rbMode || 'causal';
  const recBOn = cfg.recordB !== 'off' && cfg.recordB !== false;
  let totalNLL = 0, totalRes = 0, totalIters = 0, stepsCount = 0;
  let totalRANorm = 0, totalRBNorm = 0, totalRArBCos = 0;

  for (let start = 0; start < M - 1; start += chunkSize) {
    const end = Math.min(M - 1, start + chunkSize);
    const chunkLen = end - start;
    if (model.arch === 'transformer') {
      state = model.getEmptyState();
    } else if (model.arch === 'learns' && rbMode === 'prevrev' && recBOn && prevScoutRB) {
      state = { h: state.h, hs: state.hs, R_A: state.R_A, R_B: prevScoutRB.slice(), tokenHistory: state.tokenHistory };
    }
    for (let t = 0; t < chunkLen; t++) {
      const idx = start + t;
      const tok = splitTokens[idx];
      const target = splitTokens[idx + 1];
      const { logits, state: ns, diagnostics } = model.forward(tok, state, evalCfg);
      let mx = -Infinity;
      for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
      let sm = 0;
      for (let i = 0; i < logits.length; i++) sm += Math.exp(logits[i] - mx);
      const pT = Math.exp(logits[target] - mx) / sm;
      const stepLoss = -Math.log(pT + 1e-10);
      totalNLL += stepLoss;
      const r = roleMask ? roleMask[idx + 1] : 0;
      roleNLL[r] += stepLoss;
      roleCount[r]++;
      if (diagnostics) {
        totalRes += (diagnostics.fpResidualAvg || 0);
        totalIters += (diagnostics.fpItersAvg || 0);
        totalRANorm += (diagnostics.rANorm || 0);
        totalRBNorm += (diagnostics.rBNorm || 0);
        totalRArBCos += (diagnostics.rArBCos || 0);
      }
      stepsCount++;
      state = ns;
    }
    if (model.arch === 'learns' && rbMode === 'prevrev' && recBOn) {
      const subChunk = splitTokens.subarray ? splitTokens.subarray(start, end + 1) : splitTokens.slice(start, end + 1);
      prevScoutRB = L.scoutBackwardPass(subChunk, model, cfg).R_B_at[0].slice();
    }
  }

  const loss = totalNLL / Math.max(1, stepsCount);
  const bpb = (totalNLL / Math.LN2) / Math.max(1, stepsCount);
  return {
    loss, bpb, nll: totalNLL, count: stepsCount,
    roleNLL, roleCount,
    roleLoss: {
      other: roleCount[0] > 0 ? roleNLL[0] / roleCount[0] : null,
      step:  roleCount[1] > 0 ? roleNLL[1] / roleCount[1] : null,
      pitch: roleCount[2] > 0 ? roleNLL[2] / roleCount[2] : null
    },
    fpResidual: stepsCount ? totalRes / stepsCount : 0,
    fpIters: stepsCount ? totalIters / stepsCount : 0,
    rANorm: stepsCount ? totalRANorm / stepsCount : 0,
    rBNorm: stepsCount ? totalRBNorm / stepsCount : 0,
    rArBCos: stepsCount ? totalRArBCos / stepsCount : 0
  };
}

function evalCollectionWithRoles(items, model, cfg, chunkSize) {
  let totalNLL = 0, totalCount = 0;
  const roleNLL = [0, 0, 0];
  const roleCount = [0, 0, 0];
  const byId = {};
  let sumItemLoss = 0;

  for (const item of items) {
    const r = evalSplitWithRoles(item.tokens, item.roles, model, cfg, chunkSize);
    totalNLL += r.nll;
    totalCount += r.count;
    for (let k = 0; k < 3; k++) {
      roleNLL[k] += r.roleNLL[k];
      roleCount[k] += r.roleCount[k];
    }
    sumItemLoss += r.loss;
    byId[item.id] = {
      loss: Number(r.loss.toFixed(6)),
      bpb: Number(r.bpb.toFixed(6)),
      stepLoss: r.roleLoss.step !== null ? Number(r.roleLoss.step.toFixed(6)) : null,
      pitchLoss: r.roleLoss.pitch !== null ? Number(r.roleLoss.pitch.toFixed(6)) : null,
      otherLoss: r.roleLoss.other !== null ? Number(r.roleLoss.other.toFixed(6)) : null,
      bytes: r.count
    };
  }

  const loss = totalNLL / Math.max(1, totalCount);
  const bpb = (totalNLL / Math.LN2) / Math.max(1, totalCount);
  const itemMeanLoss = items.length > 0 ? sumItemLoss / items.length : 0;
  return {
    loss: Number(loss.toFixed(6)),
    bpb: Number(bpb.toFixed(6)),
    itemMeanLoss: Number(itemMeanLoss.toFixed(6)),
    roleLoss: {
      step:  roleCount[1] > 0 ? Number((roleNLL[1] / roleCount[1]).toFixed(6)) : null,
      pitch: roleCount[2] > 0 ? Number((roleNLL[2] / roleCount[2]).toFixed(6)) : null,
      other: roleCount[0] > 0 ? Number((roleNLL[0] / roleCount[0]).toFixed(6)) : null,
      stepBytes: roleCount[1],
      pitchBytes: roleCount[2],
      otherBytes: roleCount[0]
    },
    byId
  };
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

  const rBDiagSteps = new Set([
    1,
    Math.max(1, Math.floor(steps * 0.25)),
    Math.max(1, Math.floor(steps * 0.5)),
    Math.max(1, Math.floor(steps * 0.75)),
    steps
  ]);

  let prevStart = undefined;
  let recentTrainLosses = [];
  let firstTrainLoss = null, lastTrainLoss = null;
  let spikeCount = 0, maxBetaGrad = 0;
  let fpResidualSum = 0, fpItersSum = 0, fpAgreeSum = 0, diagCount = 0;
  let recentRANorm = [], recentRBNorm = [], recentRArBCos = [];
  const rBDiagCurve = [];
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
      const curStep = step + 1;
      if (diag) {
        fpResidualSum += (diag.fpResidualAvg || 0);
        fpItersSum += (diag.fpItersAvg || 0);
        fpAgreeSum += (diag.fpAgreeAvg || 0);
        diagCount++;
        if (diag.rANorm !== undefined) {
          recentRANorm.push(diag.rANorm); if (recentRANorm.length > 20) recentRANorm.shift();
          recentRBNorm.push(diag.rBNorm); if (recentRBNorm.length > 20) recentRBNorm.shift();
          recentRArBCos.push(diag.rArBCos); if (recentRArBCos.length > 20) recentRArBCos.shift();
          if (rBDiagSteps.has(curStep)) {
            const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
            rBDiagCurve.push({
              step: curStep,
              rANorm: Number(avg(recentRANorm).toFixed(4)),
              rBNorm: Number(avg(recentRBNorm).toFixed(4)),
              rArBCos: Number(avg(recentRArBCos).toFixed(4))
            });
          }
        }
      }

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

  const valRes = isLoomTask
    ? evalSplitWithRoles(val, valRoles, model, cfg, chunkSize)
    : L.evalFullSplit(val, model, cfg, chunkSize);
  const valLossFinal = Number(valRes.loss.toFixed(6));
  valTail.push([steps, valLossFinal]);
  if (model.arch !== 'ngram' && cli.curveEvery > 0 && steps > 0) {
    valCurve.push({ step: steps, valLoss: valLossFinal, valBpb: Number(valRes.bpb.toFixed(6)) });
  }

  let testRes = null;
  let unseenRes = null;
  let crewRes = null;
  if (cli.final) {
    if (isLoomTask) {
      unseenRes = evalCollectionWithRoles(unseenChannels, model, cfg, chunkSize);
      crewRes = evalCollectionWithRoles(crewScores, model, cfg, chunkSize);
      testRes = { loss: unseenRes.loss, bpb: unseenRes.bpb };
    } else {
      testRes = L.evalFullSplit(test, model, cfg, chunkSize);
    }
  }

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
    task: isLoomTask ? 'loom' : undefined,
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
    valRoleLoss: isLoomTask && valRes.roleLoss ? {
      step:  valRes.roleLoss.step !== null ? Number(valRes.roleLoss.step.toFixed(6)) : null,
      pitch: valRes.roleLoss.pitch !== null ? Number(valRes.roleLoss.pitch.toFixed(6)) : null,
      other: valRes.roleLoss.other !== null ? Number(valRes.roleLoss.other.toFixed(6)) : null
    } : undefined,
    valTail,
    valTailMin,
    valTailMax,
    valTailMean,
    valTailRange,
    testLoss: testRes ? Number(testRes.loss.toFixed(6)) : null,
    testBpb: testRes ? Number(testRes.bpb.toFixed(6)) : null,
    unseenTestLoss: unseenRes ? unseenRes.loss : undefined,
    unseenChannelMeanLoss: unseenRes ? unseenRes.itemMeanLoss : undefined,
    unseenRoleLoss: unseenRes ? unseenRes.roleLoss : undefined,
    unseenByChannel: unseenRes ? unseenRes.byId : undefined,
    crewTestLoss: crewRes ? crewRes.loss : undefined,
    crewScoreMeanLoss: crewRes ? crewRes.itemMeanLoss : undefined,
    crewRoleLoss: crewRes ? crewRes.roleLoss : undefined,
    crewByScore: crewRes ? crewRes.byId : undefined,
    bitsPerByte: Number((testRes ? testRes.bpb : valRes.bpb).toFixed(6)),
    wallClockSec: Number(wallSec.toFixed(2)),
    trainSec: Number(trainSec.toFixed(2)),
    trainCpuSec: Number(trainCpuSec.toFixed(2)),
    tokensPerSec: tokPerSec,
    cpuTokensPerSec: cpuTokPerSec,
    fpResidual: Number((valRes.fpResidual || (diagCount ? fpResidualSum / diagCount : 0)).toFixed(6)),
    fpItersAvg: Number((valRes.fpIters || (diagCount ? fpItersSum / diagCount : 0)).toFixed(3)),
    fpAgree: Number((diagCount ? fpAgreeSum / diagCount : 0).toFixed(6)),
    rANorm: Number((valRes.rANorm || 0).toFixed(4)),
    rBNorm: Number((valRes.rBNorm || 0).toFixed(4)),
    rArBCos: Number((valRes.rArBCos || 0).toFixed(6)),
    rBDiagCurve: rBDiagCurve.length > 0 ? rBDiagCurve : undefined,
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