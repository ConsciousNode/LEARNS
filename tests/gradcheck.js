// Full finite-difference gradient check of LEARNS v0.0013 (all Tier 2 switches including fpNorm=layernorm|none|rescale) and Tier 3 baselines (GRU, Transformer) over a 3-step sequence with BPTT.
// Usage:
//   node tests/gradcheck.js                     (default LEARNS)
//   node tests/gradcheck.js inject              (with FP input injection)
//   node tests/gradcheck.js fpNorm=rescale      (any Tier 2 switch or L12:... string)
//   node tests/gradcheck.js arch=gru            (GRU baseline)
//   node tests/gradcheck.js arch=transformer    (Transformer baseline)
//   node tests/gradcheck.js --all               (runs default, inject, every Tier 2 switch, and every baseline)
const path = require('path');
const args = process.argv.slice(2);
const htmlFile = args.find(a => a.endsWith('.html')) || path.join(__dirname, '..', 'index.html');
const L = require('./load.js')(htmlFile, { f64: true, ternary: false });

function runGradcheckFor(specStr, verbose = true) {
  L.setTernary(false);
  L.seedRNG(5);
  const V = 24, dim = 8, layers = 2;
  const parsed = L.parseConfig(specStr, {
    vocab: V, dim, layers, chunk: 8,
    fpIters: 4, innerFP: 3, fpEps: 1e-14,
    lambda: 0.3, thermo: 'linear', frontierPresent: 'on', weights: 'float32'
  });
  // If user didn't explicitly pass thermo=..., use linear (v0.0011 gradcheck default) unless thermo=hinge or thermo=off was passed
  if (!String(specStr).includes('thermo=')) parsed.thermo = 'linear';
  if (!String(specStr).includes('lambda=')) parsed.lambda = 0.3;
  if (!String(specStr).includes('fpIters=') && !String(specStr).includes('maxFPIter=')) parsed.fpIters = 4;
  if (!String(specStr).includes('innerFP=')) parsed.innerFP = 3;
  parsed.fpEps = 1e-14;

  // In float mode (default for gradcheck), check all weights; if weights=ternary is explicitly passed, check in float mode so STE weights are differentiable
  parsed.weights = 'float32';

  if (parsed.arch === 'ngram' || parsed.arch === 'unigram' || parsed.arch === 'bigram' || parsed.arch === 'trigram' || parsed.arch === '5gram') {
    if (verbose) console.log(`ALL OK (arch=${parsed.arch} has no gradient parameters)`);
    return 0;
  }

  const m = L.createModel(Object.assign({}, parsed, { vocab: V, dim, layers, K_hist: 4, hiddenDim: 8, tfDim: 8, tfFF: 16, nHeads: 2, maxCtx: 8 }));
  const bigScale = m.arch === 'gru' ? 1.5 : 15;
  const big = W => { for (let i = 0; i < W.length; i++) W[i] *= bigScale; };
  for (const k of Object.keys(m)) {
    if (/^(W_|layers_W_|b_|layers_b_)/.test(k) && !/_grad|_m1|_m2/.test(k)) {
      const v = m[k];
      if (Array.isArray(v)) v.forEach(big);
      else if (v && typeof v.length === 'number') big(v);
    }
  }
  if (m.layers_thetas) for (const T of m.layers_thetas) for (let i = 0; i < T.length; i++) T[i] = 0.3 * (i + 1);
  if (m.layers_rho) for (const R of m.layers_rho) for (let i = 0; i < R.length; i++) R[i] = 0.4 * (i % 5) - 0.6;
  if (m.layers_ln_fpA_beta) for (let l = 0; l < layers; l++) for (let i = 0; i < dim; i++) { m.layers_ln_fpA_beta[l][i] = 0.1 * (i - 3.5); m.layers_ln_fpB_beta[l][i] = 0.1 * (3.5 - i); }
  // For hinge mode, ensure dS_chunk < 0 so hinge is active and differentiable
  if (parsed.thermo === 'hinge' && m.W_rec_A) {
    for (let i = 0; i < m.W_rec_A.length; i++) m.W_rec_A[i] *= 3;
    for (let i = 0; i < m.W_rec_B.length; i++) m.W_rec_B[i] *= 3;
  }

  const cfg = Object.assign({}, parsed, { maxFPIter: parsed.fpIters, innerFPMax: parsed.innerFP, fpEps: 1e-14 });
  const seq = [3, 9, 14, 20], prefix = [1, 2];

  let s0 = m.getEmptyState();
  for (const t of prefix) s0 = m.forward(t, s0, cfg).state;

  const cloneState = s => {
    const c = {
      h: s.h ? s.h.slice() : null,
      R_A: s.R_A ? s.R_A.slice() : null,
      R_B: s.R_B ? s.R_B.slice() : null,
      tokenHistory: s.tokenHistory ? s.tokenHistory.slice() : []
    };
    if (s.hs) c.hs = s.hs.map(x => x.slice());
    if (s.kv) { c.kv = s.kv.map(x => ({ K: x.K.map(k => k.slice()), V: x.V.map(v => v.slice()) })); c.stepPos = s.stepPos || 0; }
    return c;
  };

  const recAOn = cfg.recordA !== 'off' && cfg.recordA !== false;
  const recBOn = cfg.recordB !== 'off' && cfg.recordB !== false;
  const calcH = (st) => {
    if (recAOn && recBOn) return 0.5 * (L.recordEntropy(st.R_A) + L.recordEntropy(st.R_B));
    if (recAOn) return L.recordEntropy(st.R_A);
    if (recBOn) return L.recordEntropy(st.R_B);
    return 0;
  };

  function run() {
    let st = cloneState(s0);
    const S_start = (m.arch === 'learns') ? calcH(st) : 0;
    const outs = [];
    for (let t = 0; t < seq.length - 1; t++) {
      const r = m.forward(seq[t], st, cfg);
      outs.push(r);
      st = r.state;
    }
    const S_end = (m.arch === 'learns') ? calcH(st) : 0;
    return { outs, st, dS: S_end - S_start };
  }

  function loss() {
    const { outs, dS } = run();
    let l = 0;
    outs.forEach((r, t) => { const p = L.softmax(r.logits); l -= Math.log(p[seq[t + 1]]); });
    let th = 0;
    if (m.arch === 'learns' && cfg.thermo !== 'off' && cfg.lambda && (recAOn || recBOn)) {
      th = cfg.thermo === 'hinge' ? cfg.lambda * Math.max(0, -dS) : -cfg.lambda * dS;
    }
    return l / outs.length + th;
  }

  m.zeroGrads();
  const { outs, st, dS } = run();
  const n = outs.length;
  let g = null;
  if (m.arch === 'learns' && cfg.thermo !== 'off' && cfg.lambda && (recAOn || recBOn)) {
    const active = cfg.thermo !== 'hinge' || dS < 0;
    if (active) {
      const scale = (recAOn && recBOn) ? -0.5 * cfg.lambda : -1.0 * cfg.lambda;
      const gA = recAOn ? L.entropyGradR(st.R_A) : null;
      const gB = recBOn ? L.entropyGradR(st.R_B) : null;
      if (gA) for (let j = 0; j < dim; j++) gA[j] *= scale;
      if (gB) for (let j = 0; j < dim; j++) gB[j] *= scale;
      g = { R_A: gA, R_B: gB, h: null };
    }
  }
  for (let t = n - 1; t >= 0; t--) {
    const p = Float64Array.from(L.softmax(outs[t].logits));
    p[seq[t + 1]] -= 1;
    for (let j = 0; j < p.length; j++) p[j] /= n;
    const r = m.backward(p, outs[t].caches, cfg, g);
    g = m.arch === 'learns' ? { R_A: r.grad_R_A_prev, R_B: r.grad_R_B_prev, h: r.grad_h } : r;
  }

  const eps = 1e-6;
  let fails = 0;
  function chk(name, W, G) {
    let dot = 0, na = 0, nn = 0;
    for (let i = 0; i < W.length; i++) {
      const o = W[i];
      W[i] = o + eps; const a = loss();
      W[i] = o - eps; const b = loss();
      W[i] = o;
      const num = (a - b) / (2 * eps);
      dot += G[i] * num;
      na += G[i] * G[i];
      nn += num * num;
    }
    if (nn < 1e-20 && na < 1e-20) {
      if (verbose) console.log(`${name.padEnd(20)} (no effect on loss)`);
      return;
    }
    const cos = (na > 0 && nn > 0) ? dot / Math.sqrt(na * nn) : 0;
    const ratio = (nn > 0) ? Math.sqrt(na / nn) : 0;
    const ok = cos > 0.9999 && Math.abs(ratio - 1) < 1e-3;
    if (!ok) fails++;
    if (verbose) console.log(`${name.padEnd(20)} cos=${cos.toFixed(6)} ratio=${ratio.toFixed(6)} ${ok ? 'ok' : 'FAIL'}`);
  }

  if (m.arch === 'learns') {
    const names = ['W_present', 'W_embed', 'W_rel', 'W_rec_A', 'W_rec_B', 'W_cross_A', 'W_cross_B', 'W_frontier', 'ln_recA_gamma', 'ln_recB_beta'];
    for (const k of names) chk(k, m[k], m[k + '_grad']);
    const per = ['W_A', 'W_B', 'W_gA', 'W_gB', 'W_FA', 'W_FB', 'thetas', 'rho', 'W_prog', 'W_px', 'ln_fpA_gamma', 'ln_fpB_beta', 'ln_fpAsym_gamma'];
    for (let l = 0; l < layers; l++) for (const k of per) chk(`L${l}.${k}`, m['layers_' + k][l], m['layers_' + k + '_grad'][l]);
  } else if (m.arch === 'gru') {
    chk('W_embed', m.W_embed, m.W_embed_grad);
    chk('W_out', m.W_out, m.W_out_grad);
    chk('b_out', m.b_out, m.b_out_grad);
    for (let l = 0; l < layers; l++) {
      for (const k of ['W_z', 'b_z', 'W_r', 'b_r', 'W_h', 'b_h']) {
        chk(`L${l}.${k}`, m['layers_' + k][l], m['layers_' + k + '_grad'][l]);
      }
    }
  } else if (m.arch === 'transformer') {
    chk('W_embed', m.W_embed, m.W_embed_grad);
    chk('W_out', m.W_out, m.W_out_grad);
    chk('ln_f_gamma', m.ln_f_gamma, m.ln_f_gamma_grad);
    chk('ln_f_beta', m.ln_f_beta, m.ln_f_beta_grad);
    for (let l = 0; l < layers; l++) {
      for (const k of ['W_Q', 'W_K', 'W_V', 'W_O', 'W_ff1', 'W_ff2', 'ln1_gamma', 'ln1_beta', 'ln2_gamma', 'ln2_beta']) {
        chk(`L${l}.${k}`, m['layers_' + k][l], m['layers_' + k + '_grad'][l]);
      }
    }
  }

  const label = specStr === '' ? 'without FP input injection' : (specStr === 'inject' ? 'with FP input injection' : specStr);
  console.log(fails ? `${fails} FAILED (${label})` : `ALL OK (${label})`);
  return fails;
}

if (args.includes('--all')) {
  const specs = [
    '', 'inject',
    'relEmbed=off', 'coreA=off', 'coreB=off', 'innerFP=2', 'innerFP=8',
    'fpIters=0', 'fpIters=1', 'fpIters=2', 'fpIters=6', 'fpIters=12', 'fpInject=on',
    'fpNorm=layernorm', 'fpNorm=none', 'fpNorm=rescale', 'fpNorm=none;fpInject=on', 'fpNorm=rescale;fpInject=on',
    'recordA=off', 'recordB=off', 'rbMode=prevrev',
    'frontierPresent=off', 'frontierHelix=off',
    'helix=off', 'helixRotation=none', 'helixRetention=fixed', 'helixRetention=none',
    'thermo=off', 'thermo=linear', 'thermo=hinge',
    'weights=float32', 'weights=ternary', 'clipMode=per_group', 'phiClip=off',
    'arch=gru', 'arch=transformer'
  ];
  let totalFails = 0;
  for (const s of specs) totalFails += runGradcheckFor(s, false);
  console.log(totalFails ? `TOTAL FAILED: ${totalFails}` : `ALL OK (${specs.length} configurations verified)`);
  process.exit(totalFails ? 1 : 0);
} else {
  const nonHtmlArgs = args.filter(a => !a.endsWith('.html'));
  const spec = nonHtmlArgs.join(';');
  const fails = runGradcheckFor(spec, true);
  process.exit(fails ? 1 : 0);
}