// Audit (Kehai Interim, 2026-11-04): are v0.0012's defaults bit-identical to v0.0011?
// Trains both versions for 40 steps from the same seed and chunks and compares every
// loss at full precision (all ~16 digits of the stored double, not the rounded print).
// usage: node tests/repro-v0011.js [new.html=index.html] [old.html=versions/learns-v0_0011.html]
const fs = require('fs'), vm = require('vm'), path = require('path');
const root = path.join(__dirname, '..');
const [NEW = path.join(root, 'index.html'), OLD = path.join(root, 'versions', 'learns-v0_0011.html')] = process.argv.slice(2);
function load(file) {
  const src = fs.readFileSync(file, 'utf8');
  const a = src.indexOf('function mulberry32(a) {'), b = src.lastIndexOf('/*', src.indexOf('BPE TOKENIZER'));
  const c = src.indexOf('class LEARNSModel {'), d = src.indexOf('function computeEffectiveWindow', c);
  const e = src.indexOf('function scoutBackwardPass'), f = src.indexOf('/* ── v0.0009: controls', e);
  const App = { phiClip: true };
  const ctx = { Math, Float32Array, Int8Array, WeakMap, Array, Object, console, Promise, setTimeout, App, yieldToMain: () => Promise.resolve() };
  vm.createContext(ctx);
  vm.runInContext([src.slice(a, b), src.slice(c, d), src.slice(e, f), 'this.X = { LEARNSModel, seedRNG, rand, optimizerStep, trainChunk };'].join('\n'), ctx);
  return { X: ctx.X, App };
}
(async () => {
  const toks = Array.from(fs.readFileSync(path.join(root, 'corpus.txt')));
  const runs = [];
  for (const file of [OLD, NEW]) {
    const { X, App } = load(file); X.seedRNG(3);
    const m = new X.LEARNSModel({ vocab: 256, dim: 32, layers: 2, K_hist: 32 });
    App.model = m; App.trainState = m.getEmptyState();
    const cfg = { lambda: 0.05, maxFPIter: 6, innerFPMax: 4, fpEps: 1e-4, rbMode: 'causal', contiguous: false, frontierPresent: true, thermoMode: 'hinge' };
    const L = [];
    for (let s = 1; s <= 40; s++) {
      const st = Math.floor(X.rand() * (toks.length - 66));
      L.push((await X.trainChunk(toks.slice(st, st + 65), cfg)).loss);
      X.optimizerStep(m, s, 0.004);
    }
    runs.push(L);
  }
  const same = runs[0].filter((v, i) => v === runs[1][i]).length;
  console.log(`step 1:  ${runs[0][0]} vs ${runs[1][0]}`);
  console.log(`step 40: ${runs[0][39]} vs ${runs[1][39]}`);
  console.log(same === 40 ? 'BIT-IDENTICAL (40/40 steps)' : `DIFFERENT (${same}/40 steps identical)`);
  process.exit(same === 40 ? 0 : 1);
})();
