const fs = require('fs'), vm = require('vm');
module.exports = (file, { f64 = true, ternary = true } = {}) => {
  const src = fs.readFileSync(file, 'utf8');
  const a = src.indexOf('function mulberry32(a) {');
  const b = src.lastIndexOf('/*', src.indexOf('BPE TOKENIZER'));          // start of the tokenizer banner
  const c = src.indexOf('class LEARNSModel {'), d = src.indexOf('\nfunction adamwStep', c);
  if (a < 0 || b < a || c < 0 || d < c) throw new Error('markers not found');
  const code = [src.slice(a, b), src.slice(c, d), `LEARNS_TERNARY = ${ternary};`,
    'this.L = { LEARNSModel, seedRNG, softmax, entropyGradR, recordEntropy };'].join('\n');
  const ctx = { Math, Float32Array: f64 ? Float64Array : Float32Array, Array, Object, console };
  vm.createContext(ctx); vm.runInContext(code, ctx);
  return ctx.L;
};
