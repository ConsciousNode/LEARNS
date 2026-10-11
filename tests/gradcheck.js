// Full finite-difference check of LEARNS v0.0010 (incl. helical retention ρ) over a 3-step sequence with BPTT.
// Float mode (ternary off) so every weight is checkable; float64; FP iterations fixed (eps tiny).
const L = require('./load.js')(process.argv.slice(2).find(a => a.endsWith('.html')) || require('path').join(__dirname, '..', 'index.html'), { ternary: false });
const inject = process.argv.includes('inject');
L.seedRNG(5);
const V = 24, dim = 8, layers = 2;
const m = new L.LEARNSModel({ vocab: V, dim, layers, K_hist: 4 });
// make weights big enough that nothing is degenerate
const big = W => { for (let i = 0; i < W.length; i++) W[i] *= 15; };
for (const k of Object.keys(m)) if (/^(W_|layers_W_)/.test(k) && !/_grad|_m1|_m2/.test(k)) { const v = m[k]; if (Array.isArray(v)) v.forEach(big); else big(v); }
for (const T of m.layers_thetas) for (let i = 0; i < T.length; i++) T[i] = 0.3 * (i + 1);
for (const R of m.layers_rho) for (let i = 0; i < R.length; i++) R[i] = 0.4 * (i % 5) - 0.6;   // spread σ(ρ) over 0.35..0.73
const cfg = { maxFPIter: 4, innerFPMax: 3, fpEps: 1e-14, lambda: 0.3, fpInject: inject, frontierPresent: true };
const seq = [3, 9, 14, 20], prefix = [1, 2];
let s0 = m.getEmptyState(); for (const t of prefix) s0 = m.forward(t, s0, cfg).state;
const clone = s => ({ h: s.h.slice(), R_A: s.R_A.slice(), R_B: s.R_B.slice(), tokenHistory: s.tokenHistory.slice() });
function run() { let st = clone(s0); const outs = []; for (let t = 0; t < seq.length - 1; t++) { const r = m.forward(seq[t], st, cfg); outs.push(r); st = r.state; } return { outs, st }; }
function loss() { const { outs, st } = run(); let l = 0; outs.forEach((r, t) => { const p = L.softmax(r.logits); l -= Math.log(p[seq[t + 1]]); });
  const H = R => L.recordEntropy(R); return l / outs.length - cfg.lambda * 0.5 * (H(st.R_A) + H(st.R_B)); }
// analytic, mirroring trainChunk
m.zeroGrads();
const { outs, st } = run(); const n = outs.length;
let g = null; { const gA = L.entropyGradR(st.R_A), gB = L.entropyGradR(st.R_B); for (let j = 0; j < dim; j++) { gA[j] *= -0.5 * cfg.lambda; gB[j] *= -0.5 * cfg.lambda; } g = { R_A: gA, R_B: gB, h: null }; }
for (let t = n - 1; t >= 0; t--) { const p = Float64Array.from(L.softmax(outs[t].logits)); p[seq[t + 1]] -= 1; for (let j = 0; j < p.length; j++) p[j] /= n;
  const r = m.backward(p, outs[t].caches, cfg, g); g = { R_A: r.grad_R_A_prev, R_B: r.grad_R_B_prev, h: r.grad_h }; }
const eps = 1e-6; let fails = 0;
function chk(name, W, G) { let dot = 0, na = 0, nn = 0;
  for (let i = 0; i < W.length; i++) { const o = W[i]; W[i] = o + eps; const a = loss(); W[i] = o - eps; const b = loss(); W[i] = o; const num = (a - b) / (2 * eps); dot += G[i] * num; na += G[i] * G[i]; nn += num * num; }
  if (nn < 1e-20) { console.log(`${name.padEnd(20)} (no effect on loss)`); return; }
  const cos = dot / Math.sqrt(na * nn), ratio = Math.sqrt(na / nn), ok = cos > 0.9999 && Math.abs(ratio - 1) < 1e-3; if (!ok) fails++;
  console.log(`${name.padEnd(20)} cos=${cos.toFixed(6)} ratio=${ratio.toFixed(6)} ${ok ? 'ok' : 'FAIL'}`); }
const names = ['W_present', 'W_embed', 'W_rel', 'W_rec_A', 'W_rec_B', 'W_cross_A', 'W_cross_B', 'W_frontier', 'ln_recA_gamma', 'ln_recB_beta'];
for (const k of names) chk(k, m[k], m[k + '_grad']);
const per = ['W_A', 'W_B', 'W_gA', 'W_gB', 'W_FA', 'W_FB', 'thetas', 'rho', 'W_prog', 'W_px', 'ln_fpA_gamma', 'ln_fpB_beta', 'ln_fpAsym_gamma'];
for (let l = 0; l < layers; l++) for (const k of per) chk(`L${l}.${k}`, m['layers_' + k][l], m['layers_' + k + '_grad'][l]);
console.log(fails ? `${fails} FAILED` : `ALL OK (${inject ? 'with' : 'without'} FP input injection)`);
process.exit(fails ? 1 : 0);
