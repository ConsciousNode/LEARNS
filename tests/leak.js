#!/usr/bin/env node
// Automated causality leak test & lookback window equivalence proof for LEARNS v0.0015 ("The Partner Record").
// Tests every rbMode:
//   - rbMode=causal (default)
//   - rbMode=prevrev
//   - rbMode=lookback;lookback=16
//   - rbMode=lookback;lookback=64
//   - recordB=off
//   - scoutScale=0.5;scoutHead=separate
//   - rbMode=legacy (positive control: MUST leak and fail causal invariance)
// on BOTH random weights and a trained checkpoint, across both training forward() and eval _forwardEval().
// Perturbing any byte at position p > t must leave every logit at position tau <= t bit-identical (max |Δlogit| === 0).
// Also proves that rbMode=lookback in both forward() and _forwardEval() is bit-identical (max |Δ| === 0) to an
// independent standalone scratch window recurrence over x_t, x_{t-1}, ..., x_{max(0, t-N+1)}.
// Usage:
//   node tests/leak.js
'use strict';

const fs = require('fs');
const path = require('path');

const htmlPath = path.join(__dirname, '..', 'index.html');
const L = require('./load.js')(htmlPath, { f64: false, ternary: true });

function runForwardChunk(model, prevChunk, chunkTokens, cfg, evalOnly = false) {
  const len = chunkTokens.length - 1;
  const runCfg = Object.assign({}, cfg, { evalOnly });
  const rbMode = cfg.rbMode || 'causal';
  const recBOn = cfg.recordB !== 'off' && cfg.recordB !== false;
  const legacy = rbMode === 'legacy';

  let state = model.getEmptyState();
  for (let t = 0; t < prevChunk.length - 1; t++) {
    state = model.forward(prevChunk[t], state, runCfg).state;
  }
  if (rbMode === 'prevrev' && recBOn) {
    const prevScoutRB = L.scoutBackwardPass(prevChunk, model, cfg).R_B_at[0].slice();
    state = { h: state.h.slice(), hs: state.hs ? state.hs.map(x => x.slice()) : null, R_A: state.R_A.slice(), R_B: prevScoutRB, tokenHistory: state.tokenHistory.slice() };
  } else if (rbMode === 'lookback') {
    state = model.getEmptyState();
  }

  const R_B_future = (legacy && recBOn)
    ? L.scoutBackwardPass(chunkTokens, model, cfg).R_B_at
    : null;

  const logitsAt = [];
  const rBAt = [];
  for (let t = 0; t < len; t++) {
    const stIn = legacy ? {
      h: state.h.slice(),
      hs: state.hs ? state.hs.map(x => x.slice()) : null,
      R_A: state.R_A.slice(),
      R_B: R_B_future[t].slice(),
      tokenHistory: state.tokenHistory.slice()
    } : state;
    const res = model.forward(chunkTokens[t], stIn, runCfg);
    logitsAt.push(Float64Array.from(res.logits));
    rBAt.push(Float64Array.from(res.state.R_B));
    state = legacy ? {
      h: res.state.h.slice(),
      hs: res.state.hs ? res.state.hs.map(x => x.slice()) : null,
      R_A: res.state.R_A.slice(),
      R_B: R_B_future[t + 1].slice(),
      tokenHistory: res.state.tokenHistory.slice()
    } : res.state;
  }
  return { logitsAt, rBAt };
}

function measureMaxLeak(model, prevChunk, baseChunk, cfg, evalOnly = false) {
  const len = baseChunk.length - 1;
  const baseOut = runForwardChunk(model, prevChunk, baseChunk, cfg, evalOnly).logitsAt;
  const probePositions = [0, 3, 7, 15, 23, len - 2].filter(p => p >= 0 && p < len - 1);

  let maxDelta = 0;
  for (const t of probePositions) {
    const pertTargets = new Set([t + 1, Math.min(len, t + 4), len - 1, len]);
    for (const pIdx of pertTargets) {
      if (pIdx <= t || pIdx >= baseChunk.length) continue;
      const pertChunk = Int32Array.from(baseChunk);
      pertChunk[pIdx] = (pertChunk[pIdx] + 73) & 0xff;
      const pertOut = runForwardChunk(model, prevChunk, pertChunk, cfg, evalOnly).logitsAt;
      for (let tau = 0; tau <= t; tau++) {
        const a = baseOut[tau], b = pertOut[tau];
        for (let v = 0; v < a.length; v++) {
          const d = Math.abs(a[v] - b[v]);
          if (d > maxDelta) maxDelta = d;
        }
      }
    }
  }
  return maxDelta;
}

function verifyLookbackScratchEquivalence(model, chunkTokens, cfg) {
  const len = chunkTokens.length - 1;
  const dim = model.dim;
  const lookbackN = Math.max(1, Number(cfg.lookback) || 16);

  let maxScratchTrainRBDiff = 0;
  let maxScratchEvalRBDiff = 0;

  let stTrain = model.getEmptyState();
  let stEval = model.getEmptyState();
  for (let t = 0; t < len; t++) {
    const R_A_prev_eval = stEval.R_A.slice();
    const stepTrain = model.forward(chunkTokens[t], stTrain, Object.assign({}, cfg, { evalOnly: false }));
    const stepEval = model.forward(chunkTokens[t], stEval, Object.assign({}, cfg, { evalOnly: true }));

    // 1. Standalone scratch window check against forward() (evalOnly: false)
    const K_B_train = stepTrain.caches.K_B_final;
    const R_A_prev_train = stepTrain.caches.R_A_prev;
    const cxB_train = L.matvecTernary(model.W_cross_B, R_A_prev_train, dim, dim);
    let rB_scratch_tr = L.zeros(dim);
    const M = Math.min(lookbackN, t + 1);
    for (let k = 0; k < M; k++) {
      const tok_k = chunkTokens[t - k];
      const u_k = L.zeros(dim);
      for (let j = 0; j < dim; j++) {
        u_k[j] = model.W_embed[tok_k * dim + j] + (k === 0 ? K_B_train[j] : 0);
      }
      const Kb_k = L.matvecTernary(model.W_rec_B, u_k, dim, dim);
      const sum_k = L.zeros(dim);
      for (let j = 0; j < dim; j++) sum_k[j] = rB_scratch_tr[j] + Kb_k[j] + cxB_train[j];
      rB_scratch_tr = L.layerNorm(sum_k, model.ln_recB_gamma, model.ln_recB_beta, 1e-5, false).out;
    }
    for (let j = 0; j < dim; j++) {
      maxScratchTrainRBDiff = Math.max(maxScratchTrainRBDiff, Math.abs(stepTrain.state.R_B[j] - rB_scratch_tr[j]));
    }

    // 2. Standalone scratch window check against _forwardEval() (evalOnly: true)
    const K_B_eval = model._ev.KB0;
    const cxB_eval = L.matvecTernary(model.W_cross_B, R_A_prev_eval, dim, dim);
    let rB_scratch_ev = L.zeros(dim);
    for (let k = 0; k < M; k++) {
      const tok_k = chunkTokens[t - k];
      const u_k = L.zeros(dim);
      for (let j = 0; j < dim; j++) {
        u_k[j] = model.W_embed[tok_k * dim + j] + (k === 0 ? K_B_eval[j] : 0);
      }
      const Kb_k = L.matvecTernary(model.W_rec_B, u_k, dim, dim);
      for (let j = 0; j < dim; j++) rB_scratch_ev[j] = rB_scratch_ev[j] + Kb_k[j] + cxB_eval[j];
      L.layerNorm(rB_scratch_ev, model.ln_recB_gamma, model.ln_recB_beta, 1e-5, true);
    }
    for (let j = 0; j < dim; j++) {
      maxScratchEvalRBDiff = Math.max(maxScratchEvalRBDiff, Math.abs(stepEval.state.R_B[j] - rB_scratch_ev[j]));
    }

    stTrain = stepTrain.state;
    stEval = stepEval.state;
  }

  return { maxScratchTrainRBDiff, maxScratchEvalRBDiff };
}

(async () => {
  L.Tokenizer.useByteLevel();
  const modes = [
    { name: 'rbMode=causal (B0 default)', spec: 'dim=32;layers=2;rbMode=causal;scoutScale=0', expectLeak: false },
    { name: 'recordB=off (A1)', spec: 'dim=32;layers=2;recordB=off;scoutScale=0', expectLeak: false },
    { name: 'rbMode=prevrev (A2)', spec: 'dim=32;layers=2;rbMode=prevrev;scoutScale=0', expectLeak: false },
    { name: 'rbMode=lookback;lookback=16 (A3-16)', spec: 'dim=32;layers=2;rbMode=lookback;lookback=16;scoutScale=0', expectLeak: false, isLookback: true },
    { name: 'rbMode=lookback;lookback=64 (A3-64)', spec: 'dim=32;layers=2;rbMode=lookback;lookback=64;scoutScale=0', expectLeak: false, isLookback: true },
    { name: 'scoutScale=0.5;scoutHead=separate (A4)', spec: 'dim=32;layers=2;rbMode=causal;scoutScale=0.5;scoutHead=separate', expectLeak: false },
    { name: 'rbMode=legacy (positive control)', spec: 'dim=32;layers=2;rbMode=legacy;scoutScale=0.5', expectLeak: true }
  ];

  const prevChunk = Int32Array.from({ length: 33 }, (_, i) => (65 + (i * 17) % 58) & 0xff);
  const testChunk = Int32Array.from({ length: 33 }, (_, i) => (97 + (i * 29) % 26) & 0xff);

  let failures = 0;
  console.log('=== LEARNS v0.0015 Causality Leak Test & Lookback Equivalence Suite ===');

  for (const mSpec of modes) {
    const cfg = L.parseConfig(mSpec.spec, { arch: 'learns', vocab: 256, dim: 32, layers: 2, chunk: 32 });

    // 1. Random weights check
    L.seedRNG(7);
    const modelInit = L.createModel(cfg);
    const leakInitTrain = measureMaxLeak(modelInit, prevChunk, testChunk, cfg, false);
    const leakInitEval = measureMaxLeak(modelInit, prevChunk, testChunk, cfg, true);
    const leakInit = Math.max(leakInitTrain, leakInitEval);

    // 2. Trained checkpoint check (train 15 steps so all weights, LayerNorms, and cross-couplings move)
    L.seedRNG(7);
    const modelTrained = L.createModel(cfg);
    Object.assign(L.App, {
      model: modelTrained,
      trainState: modelTrained.getEmptyState(),
      prevScoutRB: null,
      phiClip: true,
      clipMode: 'global',
      fpNorm: cfg.fpNorm || 'rescale'
    });
    for (let s = 1; s <= 15; s++) {
      const trChunk = Int32Array.from({ length: 33 }, (_, i) => (32 + ((s * 31 + i * 13) % 90)) & 0xff);
      cfg.contiguous = (s > 1);
      await L.trainChunk(trChunk, cfg);
      L.optimizerStep(modelTrained, s, 0.012, cfg);
    }
    const leakTrainedTrain = measureMaxLeak(modelTrained, prevChunk, testChunk, cfg, false);
    const leakTrainedEval = measureMaxLeak(modelTrained, prevChunk, testChunk, cfg, true);
    const leakTrained = Math.max(leakTrainedTrain, leakTrainedEval);

    if (mSpec.expectLeak) {
      const detected = (leakInit > 1e-6 && leakTrained > 1e-6);
      if (!detected) failures++;
      console.log(
        `${mSpec.name.padEnd(42)} | random max|Δlogit|=${leakInit.toExponential(3)} | trained max|Δlogit|=${leakTrained.toExponential(3)} | ` +
        `${detected ? 'LEAK DETECTED (positive control OK)' : 'FAIL (failed to detect leak!)'}`
      );
    } else {
      const ok = (leakInit === 0 && leakTrained === 0);
      if (!ok) failures++;
      let extraMsg = '';
      if (mSpec.isLookback) {
        const eqInit = verifyLookbackScratchEquivalence(modelInit, testChunk, cfg);
        const eqTrn = verifyLookbackScratchEquivalence(modelTrained, testChunk, cfg);
        const maxEqDiff = Math.max(
          eqInit.maxScratchTrainRBDiff, eqInit.maxScratchEvalRBDiff,
          eqTrn.maxScratchTrainRBDiff, eqTrn.maxScratchEvalRBDiff
        );
        if (maxEqDiff !== 0) failures++;
        extraMsg = ` | scratch-equiv max|ΔR_B|=${maxEqDiff.toExponential(2)} ${maxEqDiff === 0 ? 'OK' : 'FAIL'}`;
      }
      console.log(
        `${mSpec.name.padEnd(42)} | random max|Δlogit|=${leakInit.toExponential(3)} | trained max|Δlogit|=${leakTrained.toExponential(3)} | ` +
        `${ok ? 'PASS (0 leak)' : 'FAIL (LEAKED!)'}${extraMsg}`
      );
    }
  }

  if (failures > 0) {
    console.error(`FAILED: ${failures} causality/equivalence check(s) failed`);
    process.exit(1);
  }
  console.log('ALL OK (all causal modes have max |Δlogit| === 0 on random & trained weights; legacy positive control fails as expected; lookback scratch equivalence === 0)');
})();