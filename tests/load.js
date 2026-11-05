const fs = require('fs'), vm = require('vm');
module.exports = (file, { f64 = true, ternary = true } = {}) => {
  const src = fs.readFileSync(file, 'utf8');
  const m0 = src.indexOf('/*@@LEARNS_BEGIN*/');
  const m1 = src.indexOf('/*@@LEARNS_END*/');
  let core;
  if (m0 >= 0 && m1 > m0) {
    core = src.slice(m0 + '/*@@LEARNS_BEGIN*/'.length, m1);
  } else {
    const a = src.indexOf('function mulberry32(a) {');
    const b = src.lastIndexOf('/*', src.indexOf('BPE TOKENIZER'));
    const c = src.indexOf('class LEARNSModel {'), d = src.indexOf('\nfunction adamwStep', c);
    if (a < 0 || b < a || c < 0 || d < c) throw new Error('markers not found');
    core = [src.slice(a, b), src.slice(c, d)].join('\n');
  }
  const App = { phiClip: true };
  const code = [
    core,
    `const CODE_HASH = typeof fnv1aUTF8 === 'function' ? fnv1aUTF8(${JSON.stringify(core)}) : '0x00000000';`,
    `LEARNS_TERNARY = ${ternary};`,
    `this.L = {
      LEARNSModel,
      GRUModel: typeof GRUModel !== 'undefined' ? GRUModel : null,
      TransformerModel: typeof TransformerModel !== 'undefined' ? TransformerModel : null,
      KneserNeyModel: typeof KneserNeyModel !== 'undefined' ? KneserNeyModel : null,
      createModel: typeof createModel !== 'undefined' ? createModel : (cfg => new LEARNSModel(cfg)),
      Tokenizer: typeof Tokenizer !== 'undefined' ? Tokenizer : null,
      seedRNG, rand, randn, zeros, softmax, entropy, entropyGradR, recordEntropy,
      matvec, matvecT, matvecTernary, matvecTernaryT, layerNorm, layerNormBackward,
      rescaleNorm: typeof rescaleNorm !== 'undefined' ? rescaleNorm : null,
      rescaleNormBackward: typeof rescaleNormBackward !== 'undefined' ? rescaleNormBackward : null,
      spectralNormClip: typeof spectralNormClip !== 'undefined' ? spectralNormClip : null,
      optimizerStep: typeof optimizerStep !== 'undefined' ? optimizerStep : null,
      trainChunk: typeof trainChunk !== 'undefined' ? trainChunk : null,
      scoutBackwardPass: typeof scoutBackwardPass !== 'undefined' ? scoutBackwardPass : null,
      cosineLR: typeof cosineLR !== 'undefined' ? cosineLR : null,
      splitCorpus80_10_10: typeof splitCorpus80_10_10 !== 'undefined' ? splitCorpus80_10_10 : null,
      buildStrideStarts: typeof buildStrideStarts !== 'undefined' ? buildStrideStarts : null,
      buildSeenSegments: typeof buildSeenSegments !== 'undefined' ? buildSeenSegments : null,
      evalFullSplit: typeof evalFullSplit !== 'undefined' ? evalFullSplit : null,
      normalizeConfig: typeof normalizeConfig !== 'undefined' ? normalizeConfig : (x => x),
      parseConfig: typeof parseConfig !== 'undefined' ? parseConfig : (x => x),
      serializeConfig: typeof serializeConfig !== 'undefined' ? serializeConfig : (() => ''),
      fnv1aBytes: typeof fnv1aBytes !== 'undefined' ? fnv1aBytes : null,
      fnv1aUTF8: typeof fnv1aUTF8 !== 'undefined' ? fnv1aUTF8 : null,
      CODE_HASH,
      App,
      setTernary: (v) => { LEARNS_TERNARY = v; }
    };`
  ].join('\n');
  const ctx = {
    Math,
    Float32Array: f64 ? Float64Array : Float32Array,
    Float64Array,
    Int8Array,
    Uint8Array,
    Int32Array,
    WeakMap,
    Map,
    Array,
    Object,
    String,
    Number,
    JSON,
    TextEncoder: typeof TextEncoder !== 'undefined' ? TextEncoder : undefined,
    TextDecoder: typeof TextDecoder !== 'undefined' ? TextDecoder : undefined,
    console,
    Promise,
    setTimeout,
    App,
    yieldToMain: () => Promise.resolve()
  };
  if (!f64) {
    const fn = new Function('App', 'yieldToMain', code + '\nreturn this.L;');
    const obj = {};
    return fn.call(obj, App, () => Promise.resolve());
  }
  vm.createContext(ctx);
  vm.runInContext(code, ctx);
  return ctx.L;
};