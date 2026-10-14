// Does optimizerStep actually apply every gradient that backward() computes?
// gradcheck.js verifies the gradients themselves; this catches a parameter the optimizer skips
// (v0.0010 computed W_embed_grad but never stepped W_embed).
// usage: node tests/optimizer-coverage.js [file.html]   — exits 1 if any parameter with a nonzero
// gradient is left unchanged after a few training steps.
const fs = require('fs'), vm = require('vm'), path = require('path');
const file = process.argv.slice(2).find(a => a.endsWith('.html')) || path.join(__dirname, '..', 'index.html');
const src = fs.readFileSync(file, 'utf8');
const a = src.indexOf('function mulberry32(a) {'), b = src.lastIndexOf('/*', src.indexOf('BPE TOKENIZER'));
const c = src.indexOf('class LEARNSModel {'), d = src.indexOf('function computeEffectiveWindow', c);
const e = src.indexOf('function scoutBackwardPass'), f = src.indexOf('/* ── v0.0009: controls', e);
const App = { phiClip: true };
const ctx = { Math, Float32Array, Int8Array, WeakMap, Array, Object, console, Promise, setTimeout, App, yieldToMain: () => Promise.resolve() };
vm.createContext(ctx);
vm.runInContext([src.slice(a, b), src.slice(c, d), src.slice(e, f), 'this.X = { LEARNSModel, seedRNG, rand, optimizerStep, trainChunk };'].join('\n'), ctx);
const X = ctx.X; X.seedRNG(1);
const m = new X.LEARNSModel({ vocab: 64, dim: 16, layers: 2, K_hist: 16 });
App.model = m; App.trainState = m.getEmptyState();
const cfg = { lambda: 0.05, maxFPIter: 6, innerFPMax: 4, fpEps: 1e-4, rbMode: 'causal', contiguous: false, frontierPresent: true, thermoMode: 'hinge' };

// every <name>_grad on the model whose <name> is also present (single arrays and per-layer lists)
const flat = v => (Array.isArray(v) ? v : [v]).filter(x => x && typeof x.length === 'number');
const names = Object.keys(m).filter(k => k.endsWith('_grad') && m[k.slice(0, -5)] !== undefined).map(k => k.slice(0, -5));
const before = Object.fromEntries(names.map(n => [n, flat(m[n]).map(x => Float64Array.from(x))]));
const gradSeen = Object.fromEntries(names.map(n => [n, 0]));
const absMax = arrs => arrs.reduce((mx, x) => { for (const v of x) mx = Math.max(mx, Math.abs(v)); return mx; }, 0);

(async () => {
  for (let s = 1; s <= 5; s++) {
    const seq = Array.from({ length: 17 }, () => Math.floor(X.rand() * 64));
    await X.trainChunk(seq, cfg);
    for (const n of names) gradSeen[n] = Math.max(gradSeen[n], absMax(flat(m[n + '_grad'])));
    X.optimizerStep(m, s, 6e-3);
  }
  let bad = 0;
  for (const n of names) {
    const after = flat(m[n]);
    let moved = 0;
    after.forEach((x, i) => { for (let j = 0; j < x.length; j++) moved = Math.max(moved, Math.abs(x[j] - before[n][i][j])); });
    const skipped = gradSeen[n] > 0 && moved === 0;
    if (skipped) bad++;
    console.log(`${n.padEnd(26)} max|grad| ${gradSeen[n].toExponential(2)}  max|Δ| ${moved.toExponential(2)}  ${skipped ? 'NOT STEPPED' : 'ok'}`);
  }
  console.log(bad ? `FAIL: ${bad} parameter(s) have gradients the optimizer never applies` : `ALL OK (${names.length} parameters stepped)`);
  process.exit(bad ? 1 : 0);
})();
