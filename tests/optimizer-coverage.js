// Does optimizerStep actually apply every gradient that backward() computes?
// Verifies LEARNS v0.0016 (across all switches including rbMode=lookback, scoutHead=separate, fpNorm, helixState, scoutScale, ema) and every Tier 3 baseline (GRU, Transformer).
// Usage:
//   node tests/optimizer-coverage.js [file.html] [switch=val | arch=gru | arch=transformer | --all]
const path = require('path');
const args = process.argv.slice(2);
const file = args.find(a => a.endsWith('.html')) || path.join(__dirname, '..', 'index.html');
const L = require('./load.js')(file, { f64: false, ternary: true });

async function checkCoverageFor(specStr, verbose = true) {
  L.seedRNG(1);
  const cfg = L.parseConfig(specStr, {
    vocab: 64, dim: 16, layers: 2, chunk: 16,
    lambda: 0.05, fpIters: 6, innerFP: 4, fpEps: 1e-4,
    rbMode: 'causal', frontierPresent: 'on', thermo: 'hinge'
  });
  if (cfg.arch === 'ngram' || cfg.arch === 'unigram' || cfg.arch === 'bigram' || cfg.arch === 'trigram' || cfg.arch === '5gram') {
    if (verbose) console.log(`ALL OK (arch=${cfg.arch} has no gradient parameters)`);
    return 0;
  }

  const m = L.createModel(Object.assign({}, cfg, { vocab: 64, dim: 16, layers: 2, K_hist: 16 }));
  L.App.model = m;
  L.App.trainState = m.getEmptyState();
  L.App.prevScoutRB = null;
  L.App.phiClip = cfg.phiClip !== 'off';

  const flat = v => (Array.isArray(v) ? v : [v]).filter(x => x && typeof x.length === 'number');
  const names = Object.keys(m).filter(k => k.endsWith('_grad') && m[k.slice(0, -5)] !== undefined).map(k => k.slice(0, -5));
  const before = Object.fromEntries(names.map(n => [n, flat(m[n]).map(x => Float64Array.from(x))]));
  const gradSeen = Object.fromEntries(names.map(n => [n, 0]));
  const absMax = arrs => arrs.reduce((mx, x) => { for (const v of x) mx = Math.max(mx, Math.abs(v)); return mx; }, 0);

  for (let s = 1; s <= 5; s++) {
    const seq = Array.from({ length: 17 }, () => Math.floor(L.rand() * 64));
    cfg.contiguous = (s > 1);
    await L.trainChunk(seq, cfg);
    for (const n of names) gradSeen[n] = Math.max(gradSeen[n], absMax(flat(m[n + '_grad'])));
    L.optimizerStep(m, s, 6e-3, cfg);
  }

  let bad = 0, activeStepped = 0;
  for (const n of names) {
    const after = flat(m[n]);
    let moved = 0;
    after.forEach((x, i) => { for (let j = 0; j < x.length; j++) moved = Math.max(moved, Math.abs(x[j] - before[n][i][j])); });
    const skipped = (gradSeen[n] > 0 && moved === 0) || !Number.isFinite(gradSeen[n]) || !Number.isFinite(moved);
    if (skipped) bad++;
    if (gradSeen[n] > 0 && moved > 0) activeStepped++;
    if (verbose) {
      console.log(`${n.padEnd(26)} max|grad| ${gradSeen[n].toExponential(2)}  max|Δ| ${moved.toExponential(2)}  ${skipped ? 'NOT STEPPED' : (gradSeen[n] === 0 ? 'off (0 grad)' : 'ok')}`);
    }
  }
  const tag = specStr ? ` [${specStr}]` : '';
  console.log(bad ? `FAIL${tag}: ${bad} parameter(s) have gradients the optimizer never applies` : `ALL OK${tag} (${activeStepped}/${names.length} active parameters stepped)`);
  return bad;
}

(async () => {
  if (args.includes('--all')) {
    const specs = [
      '', 'fpNorm=layernorm;fpInject=off;scoutScale=0.5',
      'relEmbed=off', 'coreA=off', 'coreB=off', 'innerFP=2', 'innerFP=8',
      'fpIters=0', 'fpIters=1', 'fpIters=2', 'fpIters=6', 'fpIters=12', 'fpInject=off', 'fpInject=on',
      'fpNorm=layernorm', 'fpNorm=none', 'fpNorm=rescale', 'fpNorm=none;fpInject=off', 'fpNorm=rescale;fpInject=off',
      'recordA=off', 'recordB=off', 'rbMode=prevrev',
      'rbMode=lookback;lookback=16', 'rbMode=lookback;lookback=64',
      'rbMode=window;lookback=16', 'rbMode=window;lookback=64',
      'scoutScale=0', 'scoutScale=0.5', 'scoutScale=0.5;scoutHead=separate',
      'frontierPresent=off', 'frontierHelix=off',
      'helix=off', 'helixRotation=none', 'helixRetention=fixed', 'helixRetention=none', 'helixState=per_layer',
      'thermo=off', 'thermo=linear', 'thermo=hinge',
      'weights=float32', 'weights=ternary', 'clipMode=per_group', 'phiClip=off', 'ema=on',
      'arch=gru', 'arch=gru;weights=float32',
      'arch=transformer', 'arch=transformer;weights=float32'
    ];
    let totalBad = 0;
    for (const s of specs) totalBad += await checkCoverageFor(s, false);
    console.log(totalBad ? `TOTAL FAILED: ${totalBad}` : `ALL OK (${specs.length} configurations verified)`);
    process.exit(totalBad ? 1 : 0);
  } else {
    const nonHtmlArgs = args.filter(a => !a.endsWith('.html'));
    const spec = nonHtmlArgs.join(';');
    const bad = await checkCoverageFor(spec, true);
    process.exit(bad ? 1 : 0);
  }
})();