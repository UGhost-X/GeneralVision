// 临时 DOM 桩测试：用最小 document/canvas 桩运行 index.html 的完整 UI 代码，
// 覆盖 init、推理渲染、画板绘制、预置数字、步进模式、热力图、放大镜、感受野等路径。
import { readFileSync } from 'fs';

/* ---------- 桩 ---------- */
function makeCtx() {
  return {
    createImageData: (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    putImageData() {}, fillRect() {}, strokeRect() {}, beginPath() {}, moveTo() {}, lineTo() {},
    stroke() {}, setLineDash() {}, clearRect() {}, arc() {}, fill() {}, closePath() {},
  };
}
class El {
  constructor(tag = 'div') {
    this.tagName = tag; this.children = []; this.dataset = {}; this.style = {};
    this.textContent = ''; this.innerHTML = '';
    this.width = 0; this.height = 0;
    this._cls = new Set();
    this.classList = {
      add: c => this._cls.add(c), remove: c => this._cls.delete(c),
      contains: c => this._cls.has(c),
      toggle: (c, f) => { if (f === undefined) f = !this._cls.has(c); f ? this._cls.add(c) : this._cls.delete(c); return f; },
    };
    this._listeners = {};
  }
  getContext() { return makeCtx(); }
  appendChild(c) { this.children.push(c); }
  querySelector() { return new El('div'); }
  addEventListener(t, f) { (this._listeners[t] = this._listeners[t] || []).push(f); }
  setPointerCapture() {}
  getBoundingClientRect() { return { left: 0, top: 0, width: 120, height: 120, right: 120, bottom: 120 }; }
  closest() { return null; }
  fire(t, ev = {}) { ev.target = ev.target || this; for (const f of this._listeners[t] || []) f(ev); }
}

const ids = ['drawPad','inputReadout','maps-conv1','stats-conv1','maps-pool1','stats-pool1',
  'maps-conv2','stats-conv2','maps-pool2','stats-pool2','inspector','inspCanvas','inspTitle',
  'inspReadout','probBars','predBig','predConf','predSub','fcStrip','timing','rfNote',
  'btnHeat','btnStepMode','stepCtrl','stepProg','btnStepNext','outputPanel',
  'chipArch','chipParams','chipAcc','btnClear','stage-conv1','stage-pool1','stage-conv2','stage-pool2'];
const byId = new Map(ids.map(id => [id, new El('div')]));
const sampleBtns = Array.from({ length: 10 }, (_, d) => { const b = new El('button'); b.dataset.label = String(d); return b; });

global.document = {
  getElementById: id => byId.get(id) || new El('div'),
  createElement: t => new El(t),
  querySelectorAll: sel => (sel === '.sample-btn' ? sampleBtns : sel === '.stage'
    ? ['stage-conv1','stage-pool1','stage-conv2','stage-pool2'].map(id => byId.get(id)) : []),
  addEventListener: () => {},
};

/* ---------- 提取并执行页面脚本 ---------- */
const html = readFileSync('index.html', 'utf8');
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

let failures = [];
try {
  new Function(code)();   // 顶层 CORE 定义 + 立即执行的 DOM IIFE（init 在 DOMContentLoaded 注册）

  // init 注册在 DOMContentLoaded，直接手动调用
  const init = globalThis.__capture;
  // 由于 init 是闭包内函数，这里改为直接触发一次推理相关流程：
  // 实际 init 由 DOMContentLoaded 触发；我们的 document 桩里 addEventListener 是空实现，
  // 因此需要重新评估：改为在桩里捕获 DOMContentLoaded 处理器。
} catch (e) {
  failures.push('top-level eval: ' + e.message);
}

// 由于上面 document.addEventListener 是空实现，重新用可捕获的方式跑一遍
let readyHandler = null;
global.document.addEventListener = (t, f) => { if (t === 'DOMContentLoaded') readyHandler = f; };
try {
  new Function(code)();
} catch (e) {
  failures.push('top-level eval(2): ' + e.message);
}

if (!readyHandler) { failures.push('no DOMContentLoaded handler registered'); process.exit(1); }

try {
  readyHandler();   // 运行 init：buildStages/buildOutput/填芯片/加载样本7/推理渲染

  const pad = byId.get('drawPad');
  // 手绘一条笔画（pointerdown + 移动），触发节流推理
  pad.fire('pointerdown', { pointerId: 1, clientX: 50, clientY: 60, preventDefault() {} });
  pad.fire('pointermove', { pointerId: 1, clientX: 90, clientY: 100 });
  pad.fire('pointerup', { pointerId: 1 });
  await new Promise(r => setTimeout(r, 80));   // 等节流推理

  // 预置数字按钮
  sampleBtns[9].fire('click');

  // 热力图开关
  byId.get('btnHeat').fire('click', { currentTarget: byId.get('btnHeat') });

  // 分步模式 + 推进
  byId.get('btnStepMode').fire('click', { currentTarget: byId.get('btnStepMode') });
  for (let i = 0; i < 5; i++) byId.get('btnStepNext').fire('click');
  byId.get('btnStepMode').fire('click', { currentTarget: byId.get('btnStepMode') });

  // 放大镜：悬停 conv1 第 0 个特征图 → 放大画布内移动/点击 → 感受野高亮
  const conv1Grid = byId.get('maps-conv1');
  const thumb0 = conv1Grid.children[0];
  thumb0.closest = () => thumb0;  // 模拟 e.target.closest('.thumb')
  conv1Grid.fire('pointermove', { target: thumb0 });
  const insp = byId.get('inspCanvas');
  insp.fire('pointermove', { clientX: 12, clientY: 12 });
  insp.fire('pointerdown', { clientX: 12, clientY: 12 });

  // 清空
  byId.get('btnClear').fire('click');

  // 校验关键渲染结果
  const pred = byId.get('predBig').textContent;
  const stats = byId.get('stats-conv1').textContent;
  const probWidth = byId.get('probBars').children[0].children[1].style.width;
  const probRows = byId.get('probBars').children.length;
  const fcBars = byId.get('fcStrip').children.length;
  const thumbs = ['conv1','pool1','conv2','pool2'].reduce((n, k) => n + byId.get('maps-' + k).children.length, 0);
  const rfShown = byId.get('rfNote')._cls.has('show');
  const predDigit = /^[0-9]$/.test(pred);

  console.log('prediction:', pred);
  console.log('conv1 stats:', stats);
  console.log('top prob width:', probWidth);
  console.log('prob rows / fc bars / feature-map thumbs:', probRows, fcBars, thumbs);
  console.log('receptive-field note shown:', rfShown);
  console.log('prediction is a digit:', predDigit);

  if (!predDigit || probRows !== 10 || fcBars !== 64 || thumbs !== 96 || !rfShown || !/max/.test(stats)) {
    failures.push('render assertions failed');
  }
} catch (e) {
  failures.push('init/interaction: ' + e.message + '\n' + (e.stack || '').split('\n').slice(0, 3).join('\n'));
}

console.log(failures.length ? 'FAILURES:\n' + failures.join('\n') : 'DOM TEST OK');
process.exitCode = failures.length ? 1 : 0;
