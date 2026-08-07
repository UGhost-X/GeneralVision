// 临时一致性测试：index.html 内嵌的纯 JS 前向推理 vs torch 基准输出
import { readFileSync } from 'fs';

const html = readFileSync('index.html', 'utf8');
const weights = JSON.parse(readFileSync('mnist_weights.json', 'utf8'));
const expected = JSON.parse(readFileSync('expected.json', 'utf8'));

const core = html.split('// ====CORE-START====')[1].split('// ====CORE-END====')[0];
const code = 'const WEIGHTS = ' + JSON.stringify(weights) + ';\n' + core;
eval(code + '\n;globalThis.__t = { forward, W };');
const { forward } = globalThis.__t;

const MEAN = 0.1307, STD = 0.2810;
let maxErr = 0, wrongArgmax = 0;
for (const s of weights.samples) {
  const input28 = [];
  for (let y = 0; y < 28; y++) {
    const row = [];
    for (let x = 0; x < 28; x++) row.push((s.pixels[y * 28 + x] / 255 - MEAN) / STD);
    input28.push(row);
  }
  const res = forward(input28);
  const exp = expected[String(s.label)];
  const jsArg = res.probs.reduce((bi, v, i) => v > res.probs[bi] ? i : bi, 0);
  const expArg = exp.reduce((bi, v, i) => v > exp[bi] ? i : bi, 0);
  if (jsArg !== expArg) wrongArgmax++;
  let err = 0;
  for (let i = 0; i < 10; i++) err = Math.max(err, Math.abs(res.probs[i] - exp[i]));
  maxErr = Math.max(maxErr, err);
  console.log(`label ${s.label}: JS argmax=${jsArg} torch argmax=${expArg}  maxΔ=${err.toExponential(2)}`);
}
console.log('----');
console.log(`max |Δprob| across all samples: ${maxErr.toExponential(3)}`);
console.log(`argmax mismatches: ${wrongArgmax}/10`);
process.exitCode = (maxErr < 2e-3 && wrongArgmax === 0) ? 0 : 1;
