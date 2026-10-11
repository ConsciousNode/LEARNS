// Train LEARNS headless on any text file and measure whether it actually reads its input.
// usage: node tests/train-probe.js <corpus.txt> [steps=600] [seed=3]
// env:   DIM (32)  LR (6e-3)  THERMO (hinge|linear)  PRESENT (1|0)  LAMBDA (0.05)  FILE (../index.html)
// Prints causal held-out loss every 100 steps, unigram/bigram baselines for the same split,
// and a probe: how much the prediction moves when you change the CURRENT token vs the HISTORY.
const fs = require('fs'), vm = require('vm'), path = require('path');
const [corpus, STEPS = '600', SEED = '3'] = process.argv.slice(2);
if (!corpus) { console.error('usage: node tests/train-probe.js <corpus.txt> [steps] [seed]'); process.exit(2); }
const src = fs.readFileSync(process.env.FILE || path.join(__dirname, '..', 'index.html'), 'utf8');
const a = src.indexOf('function mulberry32(a) {'), b = src.lastIndexOf('/*', src.indexOf('BPE TOKENIZER'));
const c = src.indexOf('class LEARNSModel {'), d = src.indexOf('function computeEffectiveWindow', c);
const e = src.indexOf('function scoutBackwardPass'), f = src.indexOf('/* ── v0.0009: controls', e);
const App = { phiClip: true };
const ctx = { Math, Float32Array, Int8Array, WeakMap, Array, Object, console, Promise, setTimeout, App, yieldToMain: () => Promise.resolve() };
vm.createContext(ctx);
vm.runInContext([src.slice(a, b), src.slice(c, d), src.slice(e, f), 'this.X = { LEARNSModel, seedRNG, rand, softmax, optimizerStep, trainChunk };'].join('\n'), ctx);
const X = ctx.X; X.seedRNG(Number(SEED));
const toks = Array.from(new TextEncoder().encode(fs.readFileSync(corpus, 'utf8')));   // byte-level, vocab 256
const split = Math.floor(toks.length * 0.9), train = toks.slice(0, split), test = toks.slice(split, split + 300);
// baselines on the same split (add-one smoothing)
const uni = new Float64Array(256).fill(1), bi = Array.from({ length: 256 }, () => new Float64Array(256).fill(1));
for (let i = 0; i < train.length; i++) { uni[train[i]]++; if (i) bi[train[i - 1]][train[i]]++; }
const uS = uni.reduce((x, y) => x + y), bS = bi.map(r => r.reduce((x, y) => x + y));
let uL = 0, bL = 0; for (let i = 0; i < test.length - 1; i++) { uL -= Math.log(uni[test[i + 1]] / uS); bL -= Math.log(bi[test[i]][test[i + 1]] / bS[test[i]]); }
const m = new X.LEARNSModel({ vocab: 256, dim: Number(process.env.DIM || 32), layers: 2, K_hist: 32 });
App.model = m; App.trainState = m.getEmptyState();
const cfg = { lambda: Number(process.env.LAMBDA ?? 0.05), maxFPIter: 6, innerFPMax: 4, fpEps: 1e-4, rbMode: 'causal', contiguous: false,
  frontierPresent: process.env.PRESENT !== '0', thermoMode: process.env.THERMO || 'hinge' };
const heldOut = () => { let st = m.getEmptyState(), tot = 0; for (let i = 0; i < test.length - 1; i++) { const r = m.forward(test[i], st, cfg); st = r.state; tot -= Math.log(X.softmax(r.logits)[test[i + 1]] + 1e-10); } return tot / (test.length - 1); };
(async () => {
  console.log(`baselines on this split: uniform ${Math.log(256).toFixed(3)} · unigram ${(uL / (test.length - 1)).toFixed(3)} · bigram ${(bL / (test.length - 1)).toFixed(3)}`);
  const t0 = Date.now(); console.log(`step 0: held-out ${heldOut().toFixed(3)}`);
  for (let s = 1; s <= Number(STEPS); s++) {
    const st = Math.floor(X.rand() * (train.length - 34));
    await X.trainChunk(train.slice(st, st + 33), cfg);
    X.optimizerStep(m, s, Number(process.env.LR ?? 6e-3));
    if (s % 100 === 0) console.log(`step ${s}: held-out ${heldOut().toFixed(3)}  (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  }
  const KL = (p, q) => { let k = 0; for (let i = 0; i < p.length; i++) k += p[i] * Math.log((p[i] + 1e-12) / (q[i] + 1e-12)); return k; };
  let dTok = 0, dHist = 0, n1 = 0, n2 = 0; const states = [];
  { let st = m.getEmptyState(); for (let i = 0; i < 200 && i < test.length; i++) { st = m.forward(test[i], st, cfg).state; if (i % 20 === 19) states.push(st); } }
  const probe = [32, 101, 116, 97, 111, 110, 115, 105];   // space e t a o n s i
  for (const st of states) { const ps = probe.map(t => X.softmax(m.forward(t, st, cfg).logits)); for (let i = 0; i < ps.length; i++) for (let j = i + 1; j < ps.length; j++) { dTok += KL(ps[i], ps[j]); n1++; } }
  for (const t of probe) { const ps = states.map(st => X.softmax(m.forward(t, st, cfg).logits)); for (let i = 0; i < ps.length; i++) for (let j = i + 1; j < ps.length; j++) { dHist += KL(ps[i], ps[j]); n2++; } }
  console.log(`probe: change CURRENT token → ${(dTok / n1).toExponential(2)} nats · change HISTORY → ${(dHist / n2).toExponential(2)} nats`);
})();
