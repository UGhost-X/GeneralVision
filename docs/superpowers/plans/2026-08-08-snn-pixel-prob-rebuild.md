# SNN 演示页：像素→概率 实时演算 + 参数更新可视化 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `snn_demo.html` 整页精简重做——删除工作流式可视化（4 阶段 pipeline、栅格图、示波器、静态流程图），改为两块聚焦面板：**像素→概率 实时演算**（当前图的真实数值代入 6 步公式）与**参数更新**（权重网格 + STDP ΔW 实时高亮）。

**Architecture:** 单文件 `snn_demo.html`（HTML+CSS+JS）。保留控制条、输入图、概率柱、权重网格与全部仿真核心逻辑；`simStep()/fire()/resetWeights()` 数值算法不变，只加数据埋点（`preCount`/`flash`/放电快照）。仿真同时驱动两块新面板。

**Tech Stack:** 原生 HTML5 + Canvas 2D + ES6（无依赖、无构建、无测试框架；验证靠浏览器人工检查 + `node --check` 语法校验）。

## Global Constraints

- 唯一改动文件：`snn_demo.html`（其他文件不动）。
- **CLAUDE.md 工作流**：修改代码前必须先 `git add -A && git commit && git push origin main`（Task 1 执行，含当前未追踪文件）。
- 数值常量逐字保留，不得改动：`SPIKE_GAIN=0.6`、`LEAK=0.94`、`T=200`、`A_PLUS=0.8`、`W_NORM=78.4`、`THETA0=15`、`TAU_PLUS=3`、`RATE_ALPHA=0.002`、`BETA=400`。
- 不改 `simStep()` / `fire()` / `resetWeights()` 的数值算法（只加埋点）。
- 深色主题 CSS 变量（`--bg/--panel/--panel-2/--rule/--rule-soft/--ink/--ink-dim/--ink-faint/--amber/--cyan/--red/--green/--mono`）全部保留，不新增色值。
- 界面文案用中文；commit message 用中文描述。
- 设计规格：`docs/superpowers/specs/2026-08-08-snn-pixel-prob-rebuild-design.md`。

---

### Task 1: 前置提交未追踪文件（CLAUDE.md 工作流要求）

**Files:**
- Commit: `snn.py`、`data_loading.py`、`_exp_*.py`、`_bench.py`、`_diag.py`、`_prof.py`、`_verify_single.py` 等（当前 `git status` 的 `??` 项）

**Interfaces:**
- Consumes: 无
- Produces: 干净的 git 工作区（后续任务只含 `snn_demo.html` 的改动）

- [ ] **Step 1: 确认待提交文件清单**

Run: `git status --short`
Expected: 出现 `_bench.py`、`_diag.py`、`_exp_*.py`、`_prof.py`、`_verify_single.py`、`_final_baseline.py`、`data_loading.py`、`snn.py`（`??` 未追踪）。

- [ ] **Step 2: 提交并推送全部未追踪文件**

```bash
git add -A
git commit -m "chore: 提交未追踪的 SNN 进化实验脚本与数据加载模块

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push origin main
```

- [ ] **Step 3: 确认推送成功**

Run: `git status --short`
Expected: 干净（无 `??` 项）。若提交前文件有更新，则只剩 `snn_demo.html` 相关改动。

---

### Task 2: 精简 HTML 结构

**Files:**
- Modify: `snn_demo.html:322-327`（标题区）、`362-454`（pipeline+legend+scope → 换成 io-row+calc-panel）、`456-474`（learn-panel → 参数更新面板）、`476-537`（删除 flow-panel）

**Interfaces:**
- Consumes: 无
- Produces: 新 DOM 元素 id，供 Task 5/6/7 使用：
  - `#calc1`…`#calc6`（演算面板 6 行数值 span）
  - `#calcSpark`（③ 行迷你膜电位 canvas，160×36）
  - `#dwReadout`（ΔW 读出行）
  - 保留 `#canvasIn`/`#inputLabel`/`#canvasProb`/`#verdictMain`/`#verdictSub`/`#canvasGrid`/`#learnCount`/`#progBar`/`#legend`

- [ ] **Step 1: 替换标题区（322–327）**

旧：
```html
  <p class="eyebrow">Spike &middot; Integrate &middot; Fire &middot; Learn</p>
  <h1>模拟生物神经元：一个脉冲神经网络的工作流程</h1>
  <p class="subtitle">
    输入在这里不是连续的<b>数值</b>，而是一串串全或无的<b>放电脉冲</b>。网络没有损失函数、没有反向传播——
    突触权重只按放电的<b>先后时序</b>调整（STDP）。观看这个 784→100 的单层脉冲网络，从噪声开始，自己长出偏好数字的神经元。
  </p>
```
新：
```html
  <p class="eyebrow">Spike &middot; Integrate &middot; Fire &middot; Learn</p>
  <h1>像素 → 概率：这张图是怎么算出来的</h1>
  <p class="subtitle">
    一张 <b>28×28</b> 灰度图进入 784→100 的单层 LIF 脉冲网络，怎么变成 0–9 的<b>概率</b>？
    下方 6 步用当前这张图的<b>真实数值</b>实时演算；学习模式下，STDP 逐次<b>更新权重</b>——网格上闪白的位置就是刚被增强的突触。
  </p>
```

- [ ] **Step 2: 用「输入→输出 + 实时演算面板」替换 pipeline 与 scope（362–454）**

删除 `<!-- pipeline -->` 注释、`<div class="pipeline">…</div>`（363–432）、`<div class="pipeline-legend">…</div>`（434–438）、`<!-- scope -->` 注释、`<div class="scope-grid">…</div>`（441–454），原位插入：
```html
  <!-- ================= input → output ================= -->
  <div class="io-row">
    <div class="io-card">
      <p class="io-title">输入图像</p>
      <div class="canvas-host pxframe">
        <canvas id="canvasIn" width="28" height="28" class="px" style="width:132px;height:132px"
                role="img" aria-label="当前输入的手写数字图像"></canvas>
        <span class="badge" id="inputLabel">—</span>
      </div>
    </div>
    <div class="io-arrow" aria-hidden="true">→</div>
    <div class="io-card io-out">
      <p class="io-title">输出概率</p>
      <div class="verdict">
        <div class="verdict-main" id="verdictMain">—</div>
        <div class="verdict-sub" id="verdictSub">请先按 ▶ 播放，或切到「学习」</div>
      </div>
      <div class="canvas-host">
        <canvas id="canvasProb" width="360" height="110" class="line" role="img"
                aria-label="0到9数字的分类概率柱状图"></canvas>
      </div>
    </div>
  </div>

  <!-- ================= pixel→probability: live calculation ================= -->
  <section class="calc-panel" aria-label="从像素到概率的实时数值演算">
    <p class="panel-title">像素 → 概率：这张图是怎么算出来的</p>
    <p class="panel-sub">按 ▶ 播放。下面 6 步用当前这张图的真实数值逐行演算，样本放完 ⑥ 给出最终概率。</p>
    <div class="calc-rows">
      <div class="calc-row"><span class="cno">①</span><code class="cf">px[i] ∈ [0,1] · 784 维</code><span class="cv" id="calc1">—</span></div>
      <div class="calc-row"><span class="cno">②</span><code class="cf">P = px × 0.6</code><span class="cv" id="calc2">—</span></div>
      <div class="calc-row"><span class="cno">③</span><code class="cf">V ← 0.94·V + Σ W·spike · V ≥ θ</code><span class="cv" id="calc3">—</span><canvas id="calcSpark" width="160" height="36" class="spark" aria-hidden="true"></canvas></div>
      <div class="calc-row"><span class="cno">④</span><code class="cf">spikeAccum[j]++ · T = 200</code><span class="cv" id="calc4">—</span></div>
      <div class="calc-row"><span class="cno">⑤</span><code class="cf">c_d = Σ_{pref[j]=d} spikeAccum[j]</code><span class="cv" id="calc5">—</span></div>
      <div class="calc-row"><span class="cno">⑥</span><code class="cf">P(d) = c_d / Σ</code><span class="cv" id="calc6">—</span></div>
    </div>
  </section>
```

- [ ] **Step 3: 把 learn-panel 改造成「参数更新面板」（456–474）**

旧（整块）：
```html
  <!-- ================= learning ================= -->
  <div class="learn-panel">
    <div class="learn-head">
      <p class="learn-title">学习 <em>STDP</em> · 神经元自组织</p>
      <p class="learn-count">已学习 <b id="learnCount">0</b> / 400 个样本</p>
    </div>
    <div class="learn-body">
      <div class="canvas-host">
        <canvas id="canvasGrid" width="336" height="336" class="px"
                style="width:min(336px,100%)" role="img" aria-label="100个神经元学到的权重，按偏好数字着色"></canvas>
      </div>
      <p class="learn-cap">
        这 100 个神经元从<b>随机噪声</b>出发，仅靠放电时序自组织出偏好——
        颜色 = 它最常为哪个数字放电。下方是<b>像素 → 概率</b>的计算流程图。
      </p>
    </div>
    <div class="legend" id="legend"></div>
    <div class="prog"><i id="progBar"></i></div>
  </div>
```
新：
```html
  <!-- ================= parameter update (STDP ΔW) ================= -->
  <section class="learn-panel dw-panel" aria-label="参数更新可视化">
    <div class="learn-head">
      <p class="learn-title">参数更新 <em>STDP</em> · 实时 ΔW</p>
      <p class="learn-count">已学习 <b id="learnCount">0</b> / 400 个样本</p>
    </div>
    <div class="learn-body">
      <div class="canvas-host">
        <canvas id="canvasGrid" width="336" height="336" class="px"
                style="width:min(336px,100%)" role="img" aria-label="100个神经元学到的权重，按偏好数字着色"></canvas>
      </div>
      <p class="learn-cap">
        这 100 个神经元从<b>随机噪声</b>出发，仅靠放电时序自组织出偏好——
        颜色 = 它最常为哪个数字放电。<b>闪白</b>的像素 = 本次放电被 <code>+A_PLUS</code> 增强的突触。
      </p>
    </div>
    <p class="dw-readout" id="dwReadout">播放一轮「学习」，观察权重如何被 STDP 逐次更新。</p>
    <div class="legend" id="legend"></div>
    <div class="prog"><i id="progBar"></i></div>
  </section>
```

- [ ] **Step 4: 删除 flow-panel 区块（476–537）**

删除 `<!-- ================= pixel→probability flow ================= -->` 到 `</section>`（含 `.flow-panel` 内联 SVG 6 节点）整块。`<p class="note">`（539–543）保留不动。

- [ ] **Step 5: 浏览器快照检查**

打开 `snn_demo.html`：旧 4 阶段卡片 / 栅格图 / 示波器 / 静态流程图消失；新 io-row + calc-panel（6 行「—」）+ 参数更新面板出现；页面能正常滚动、无 JS 报错（旧 JS 仍引用已删除元素会报 null——本步骤只看结构，Task 7 接线后消除）。

- [ ] **Step 6: Commit**

```bash
git add snn_demo.html
git commit -m "refactor: 精简 HTML 结构——删除 pipeline/scope/flow，新增 io-row/calc-panel/参数更新面板

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: CSS 精简与新面板样式

**Files:**
- Modify: `snn_demo.html` CSS 区（删除 120–157 的 `.pipeline/.stage*`、195–224 的 `.conn/.pipeline-legend/.sw`、227–243 的 `.scope-*`、293–307 的 `.flow-*`、309–316 的媒体查询；保留 158–193 的 `.canvas-host/canvas.px/canvas.line/.pxframe/.badge/.verdict*/.learn-note`；新增 io/calc/dw 样式）

**Interfaces:**
- Consumes: Task 2 的 HTML 类名（`.io-row/.io-card/.io-arrow/.io-title/.calc-panel/.panel-title/.panel-sub/.calc-rows/.calc-row/.cno/.cf/.cv/.spark/.dw-readout`）
- Produces: 供验证的完整视觉布局

- [ ] **Step 1: 删除 pipeline 与 stage 样式（119–157），原位加入新样式**

旧（`/* ---------- pipeline ---------- */` 到 `.stage-cap b { … }`）替换为：
```css
  /* ---------- input → output ---------- */
  .io-row { display: flex; align-items: stretch; gap: 22px; margin: 22px 0 26px; }
  .io-card {
    background: var(--panel);
    border: 1px solid var(--rule);
    border-radius: 8px;
    padding: 16px 18px 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
  }
  .io-card.io-out { flex: 1 1 auto; min-width: 0; }
  .io-title {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-dim);
    margin: 0;
  }
  .io-arrow { align-self: center; font-size: 26px; color: var(--ink-faint); font-family: var(--mono); }

  /* ---------- live calculation panel ---------- */
  .calc-panel {
    margin: 26px 0 0;
    padding: 20px 20px 18px;
    background: linear-gradient(180deg, #121A2B, var(--panel));
    border: 1px solid var(--rule);
    border-radius: 8px;
  }
  .panel-title { font-family: var(--mono); font-size: 17px; font-weight: 600; margin: 0 0 6px; }
  .panel-sub { color: var(--ink-dim); font-size: 13px; margin: 0 0 16px; max-width: 70ch; }
  .calc-rows { display: flex; flex-direction: column; gap: 8px; }
  .calc-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    background: var(--panel-2);
    border: 1px solid var(--rule-soft);
    border-radius: 6px;
    flex-wrap: wrap;
  }
  .cno { font-family: var(--mono); font-weight: 700; color: var(--amber); }
  .cf { font-family: var(--mono); font-size: 12.5px; color: var(--green); white-space: nowrap; }
  .cv { font-family: var(--mono); font-size: 12.5px; color: var(--cyan); min-width: 0; }
  .spark { display: block; margin-left: auto; }
  .dw-readout { margin: 12px 0 0; font-family: var(--mono); font-size: 12.5px; color: var(--ink-dim); }
  .dw-readout b { color: var(--amber); }
```

- [ ] **Step 2: 删除连接符与图例样式（195–224）**

删除整块：`/* connectors */` 的 `.conn` 全部规则、`@keyframes dashmove`、`.conn-label`、`.pipeline-legend` 全部规则、`.sw`/`.sw.amber/.sw.cyan/.sw.red`。

- [ ] **Step 3: 删除 scope 样式（227–243）**

删除整块：`/* ---------- scope panel ---------- */` 到 `.scope-cap { … }`。

- [ ] **Step 4: 删除 flow 样式（293–307）**

删除整块：`/* ---------- pixel→probability flow ---------- */` 到 `.flow-arrow { … }`。

- [ ] **Step 5: 更新媒体查询（309–316）**

旧：
```css
  @media (max-width: 900px) {
    .pipeline { grid-template-columns: 1fr; }
    .conn svg { transform: rotate(90deg); }
    .scope-grid { grid-template-columns: 1fr; }
  }
  @media (prefers-reduced-motion: reduce) {
    .conn .cline { animation: none; }
  }
```
新：
```css
  @media (max-width: 900px) {
    .io-row { flex-direction: column; align-items: stretch; }
    .io-arrow { transform: rotate(90deg); }
  }
```

- [ ] **Step 6: 浏览器检查布局**

打开 `snn_demo.html`：io-row 横排、calc-panel 6 行底色块整齐、参数更新面板与 legend/prog 正常；900px 窄窗口 io-row 变纵向；无水平滚动。

- [ ] **Step 7: Commit**

```bash
git add snn_demo.html
git commit -m "style: 新增 io/calc/dw 面板样式，移除 pipeline/conn/scope/flow 样式

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: JS 数据埋点（preCount / flash / 放电快照）

**Files:**
- Modify: `snn_demo.html:596`（状态变量）、`598-609`（resetWeights）、`624-628`（startSample）、`636-670`（simStep）、`672-691`（fire）

**Interfaces:**
- Consumes: 无（纯新增状态）
- Produces:
  - `preCount: Float32Array(NIN)`——每个输入像素本样本累计放电次数（Task 5 的 ② 用）
  - `flash: Float32Array(NIN*N)`——ΔW 高亮强度，`fire()` 置 1，逐帧衰减（Task 6 用）
  - `fireT/fireV/fireTheta`——最近一次放电时刻/V/θ 快照（Task 5 的 ③ 用）
  - `lastLTP/lastNorm`——最近一次放电的 LTP 突触数 / 归一化系数（Task 6 的读出用）

- [ ] **Step 1: 新增状态变量（596 行 `const encBuf…` 之后）**

在 `const encBuf = new Float32Array(NIN);` 后追加：
```js
let preCount = new Float32Array(NIN);          // per-input-pixel spike count (current sample)
let flash = new Float32Array(NIN * N);         // ΔW highlight intensity, decays per frame
let fireT = -1, fireV = 0, fireTheta = THETA0; // snapshot of the last firing event
let lastLTP = 0, lastNorm = 1;                 // last firing: LTP synapse count + norm factor
```

- [ ] **Step 2: resetWeights 重置新状态（606–608）**

旧：
```js
  V.fill(0); theta.fill(THETA0); refr.fill(0); rate.fill(0);
  digCount.fill(0); pref.fill(0); spikeAccum.fill(0); encBuf.fill(0);
  lastPre.fill(-1); trained = 0; lastFire = -1; cur = null;
```
新：
```js
  V.fill(0); theta.fill(THETA0); refr.fill(0); rate.fill(0);
  digCount.fill(0); pref.fill(0); spikeAccum.fill(0); encBuf.fill(0);
  preCount.fill(0); flash.fill(0);
  fireT = -1; fireV = 0; fireTheta = THETA0; lastLTP = 0; lastNorm = 1;
  lastPre.fill(-1); trained = 0; lastFire = -1; cur = null;
```

- [ ] **Step 3: startSample 重置新状态（624–628）**

旧：
```js
function startSample(s) {
  cur = s; t = 0; lastFire = -1;
  spikeAccum.fill(0); fireHist.fill(-1); lastPre.fill(-1); V.fill(0); refr.fill(0);
  setInputLabel(s);
}
```
新：
```js
function startSample(s) {
  cur = s; t = 0; lastFire = -1;
  spikeAccum.fill(0); fireHist.fill(-1); lastPre.fill(-1); V.fill(0); refr.fill(0);
  preCount.fill(0); flash.fill(0);
  setInputLabel(s);
}
```

- [ ] **Step 4: simStep 累计像素放电次数（642–648）**

旧：
```js
  for (let k = 0; k < spikeList.length; k++) {
    const i = spikeList[k];
    lastPre[i] = t;
    encBuf[i] = 1;
    const base = i * N;
    for (let j = 0; j < N; j++) if (refr[j] <= 0) V[j] += W[base + j];
  }
```
新（只加一行 `preCount[i]++;`）：
```js
  for (let k = 0; k < spikeList.length; k++) {
    const i = spikeList[k];
    lastPre[i] = t;
    encBuf[i] = 1;
    preCount[i]++;
    const base = i * N;
    for (let j = 0; j < N; j++) if (refr[j] <= 0) V[j] += W[base + j];
  }
```

- [ ] **Step 5: simStep 捕获放电快照（656）**

旧：
```js
  if (winner >= 0) fire(winner);
```
新：
```js
  if (winner >= 0) {
    fireT = t; fireV = bestV; fireTheta = theta[winner];
    fire(winner);
  }
```

- [ ] **Step 6: fire 统计 LTP 并置 flash（679–686）**

旧：
```js
  refr[j] = REFR;
  for (let k = 0; k < spikeList.length; k++) {
    const i = spikeList[k];
    if (lastPre[i] >= 0 && t - lastPre[i] <= TAU_PLUS) W[i * N + j] += A_PLUS;
  }
  let s = 0;                                       // weight normalization
  for (let i = 0; i < NIN; i++) s += W[i * N + j];
  if (s > 0) { const c = W_NORM / s; for (let i = 0; i < NIN; i++) W[i * N + j] *= c; }
```
新：
```js
  refr[j] = REFR;
  let ltp = 0;                                     // count of LTP-updated synapses this fire
  for (let k = 0; k < spikeList.length; k++) {
    const i = spikeList[k];
    if (lastPre[i] >= 0 && t - lastPre[i] <= TAU_PLUS) {
      W[i * N + j] += A_PLUS;
      if (mode === "learn") flash[i * N + j] = 1;
      ltp++;
    }
  }
  let s = 0;                                       // weight normalization
  for (let i = 0; i < NIN; i++) s += W[i * N + j];
  let c = 1;
  if (s > 0) { c = W_NORM / s; for (let i = 0; i < NIN; i++) W[i * N + j] *= c; }
  lastLTP = ltp; lastNorm = c;
```

- [ ] **Step 7: 语法校验**

Run:
```bash
awk '/<script>/{f=1; next} /<\/script>/{f=0} f' snn_demo.html > _check.js && node --check _check.js && rm _check.js
```
Expected: 无输出、退出码 0（语法合法）。

- [ ] **Step 8: Commit**

```bash
git add snn_demo.html
git commit -m "feat: 新增 preCount/flash/放电快照/ΔW统计 数据埋点

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: 实时演算面板渲染（renderCalc）

**Files:**
- Modify: `snn_demo.html:883-888`（computeProb 拆出 computeCounts）、`renderProb` 之后新增 `setCalc/renderCalc/renderCalcSpark`

**Interfaces:**
- Consumes: Task 4 的 `preCount/fireT/fireV/fireTheta/lastFire`、`cur.px/V/theta/spikeAccum/pref`、`vAll/thAll/fireHist`
- Produces: `renderCalc()`、`renderCalcSpark(j)`、`setCalc(id,text)`、`computeCounts()`（供 Task 6/7 与 renderStage4 复用）

- [ ] **Step 1: 拆出 computeCounts（883–888）**

旧：
```js
function computeProb() {
  const counts = new Array(10).fill(0);
  let total = 0;
  for (let j = 0; j < N; j++) { counts[pref[j]] += spikeAccum[j]; total += spikeAccum[j]; }
  return total > 0 ? counts.map(c => c / total) : null;
}
```
新：
```js
function computeCounts() {
  const counts = new Array(10).fill(0);
  let total = 0;
  for (let j = 0; j < N; j++) { counts[pref[j]] += spikeAccum[j]; total += spikeAccum[j]; }
  return { counts, total };
}
function computeProb() {
  const { counts, total } = computeCounts();
  return total > 0 ? counts.map(c => c / total) : null;
}
```

- [ ] **Step 2: 新增演算面板渲染函数（`renderProb` 之后插入）**

```js
// ---------- live calculation panel ----------
function setCalc(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}
function renderCalc() {
  if (!cur) {
    for (let k = 1; k <= 6; k++) setCalc("calc" + k, "—");
    renderCalcSpark(null);
    return;
  }
  let mi = 0;                                      // brightest pixel
  for (let i = 1; i < NIN; i++) if (cur.px[i] > cur.px[mi]) mi = i;
  let bright = 0;
  for (let i = 0; i < NIN; i++) if (cur.px[i] > 0.5) bright++;
  setCalc("calc1", "最亮像素 px=" + cur.px[mi].toFixed(2) + " · 亮像素(px>0.5) " + bright + " 个");

  const P = cur.px[mi] * SPIKE_GAIN;
  setCalc("calc2", "最亮像素 P=" + P.toFixed(2) + " · 该像素本图已放电 " + Math.round(preCount[mi]) + "/" + t + " 步");

  if (lastFire >= 0 && fireT >= 0) {
    setCalc("calc3", "神经元 #" + lastFire + " 在 t=" + fireT + " 步 V=" + fireV.toFixed(1) + " ≥ θ=" + fireTheta.toFixed(1) + " 放电");
  } else {
    setCalc("calc3", "尚无神经元放电");
  }

  let total = 0;
  for (let j = 0; j < N; j++) total += spikeAccum[j];
  setCalc("calc4", (lastFire >= 0 ? "神经元 #" + lastFire + " 本图放电 " + Math.round(spikeAccum[lastFire]) + " 次" : "尚未放电") + " · 全体共 " + Math.round(total) + " 次");

  if (mode === "learn") {
    setCalc("calc5", "学习模式下不判定概率");
    setCalc("calc6", "—");
  } else if (trained === 0) {
    setCalc("calc5", "先切「学习」播放一轮，才有神经元偏好分组");
    setCalc("calc6", "—");
  } else {
    const { counts } = computeCounts();
    setCalc("calc5", "c_d = " + counts.join(" / "));
    const prob = computeProb();
    if (prob) {
      const maxD = argmax(prob, 10);
      setCalc("calc6", "P = " + prob.map(x => x.toFixed(2)).join(" / ") + "  → 最可能数字 " + maxD);
    } else {
      setCalc("calc6", "本图无神经元放电");
    }
  }
  renderCalcSpark(lastFire >= 0 ? lastFire : null);
}
function renderCalcSpark(j) {
  const cv = document.getElementById("calcSpark");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (j === null || j < 0 || mode !== "infer" || t === 0) return;
  const w = cv.width, h = cv.height;
  const xScale = w / T;
  let vmax = 0;
  for (let tt = 0; tt < t; tt++) {
    const v = vAll[j * T + tt] / thAll[j * T + tt];
    if (v > vmax) vmax = v;
  }
  if (vmax <= 0) return;
  ctx.beginPath();
  for (let tt = 0; tt < t; tt++) {
    const nv = vAll[j * T + tt] / thAll[j * T + tt] / vmax;
    const x = (tt + 0.5) * xScale;
    const y = h - 5 - nv * (h - 9);
    if (tt === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = "#4CC9F0"; ctx.lineWidth = 1.4; ctx.stroke();
  for (let tt = 0; tt < t; tt++) {
    if (fireHist[tt] !== j) continue;
    const nv = vAll[j * T + tt] / thAll[j * T + tt] / vmax;
    const x = (tt + 0.5) * xScale;
    const y = h - 5 - nv * (h - 9);
    ctx.fillStyle = "#FFB454";
    ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI * 2); ctx.fill();
  }
}
```

- [ ] **Step 3: 语法校验**

Run:
```bash
awk '/<script>/{f=1; next} /<\/script>/{f=0} f' snn_demo.html > _check.js && node --check _check.js && rm _check.js
```
Expected: 无输出、退出码 0。

- [ ] **Step 4: 浏览器初验**

打开页面，切「推理」播放：① 最亮像素/亮像素数随图变化；② 放电计数随播放增长；③ 出现「神经元 #x 在 t=… V=… ≥ θ=… 放电」且右侧 sparkline 青色曲线 + 琥珀放电点；④ 计数正确；⑤⑥ 未学习时显示提示。

- [ ] **Step 5: Commit**

```bash
git add snn_demo.html
git commit -m "feat: 实时演算面板 renderCalc（6 步真实数值 + 迷你膜电位 sparkline）

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: 参数更新面板（renderDw + ΔW 高亮）

**Files:**
- Modify: `snn_demo.html:851-877`（renderGrid 叠加 flash）、`renderCalc` 之后新增 `decayFlash/renderDw`

**Interfaces:**
- Consumes: Task 4 的 `flash/lastLTP/lastNorm/lastFire/pref`、`A_PLUS/W_NORM`
- Produces: `decayFlash()`、`renderDw()`（供 Task 7 接线）

- [ ] **Step 1: renderGrid 叠加 flash 白亮（869–874）**

旧：
```js
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
      const i = x + y * 28;
      const wv = W[i * N + n] / maxW;               // W stored as [pixel*N + neuron]
      const o = ((oy + y) * Wg + (ox + x)) * 4;
      d[o] = c[0] * wv; d[o + 1] = c[1] * wv; d[o + 2] = c[2] * wv; d[o + 3] = 255;
    }
```
新：
```js
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
      const i = x + y * 28;
      const wv = W[i * N + n] / maxW;               // W stored as [pixel*N + neuron]
      const f = flash[i * N + n];                   // ΔW highlight: blend toward white
      const o = ((oy + y) * Wg + (ox + x)) * 4;
      d[o] = c[0] * wv + 255 * f; d[o + 1] = c[1] * wv + 255 * f;
      d[o + 2] = c[2] * wv + 255 * f; d[o + 3] = 255;
    }
```

- [ ] **Step 2: 新增 decayFlash + renderDw（`renderCalcSpark` 之后插入）**

```js
// ---------- parameter update (ΔW) ----------
function decayFlash() {
  for (let p = 0; p < NIN * N; p++) flash[p] *= 0.8;
}
function renderDw() {
  const el = document.getElementById("dwReadout");
  if (!el) return;
  if (mode !== "learn") {
    el.textContent = "学习模式下，放电神经元被增强的突触会闪白。先切到「学习」播放。";
    return;
  }
  if (lastFire < 0) {
    el.textContent = "播放一轮「学习」，观察权重如何被 STDP 逐次更新。";
    return;
  }
  el.textContent = "刚放电 神经元 #" + lastFire + "（偏好数字 " + pref[lastFire] + "）："
    + lastLTP + " 个突触 +" + A_PLUS.toFixed(1) + "，整列归一化 ×" + lastNorm.toFixed(2);
}
```

- [ ] **Step 3: 语法校验**

Run:
```bash
awk '/<script>/{f=1; next} /<\/script>/{f=0} f' snn_demo.html > _check.js && node --check _check.js && rm _check.js
```
Expected: 无输出、退出码 0。

- [ ] **Step 4: 浏览器初验**

切「学习」播放：网格上每次放电对应神经元块的被增强像素**闪白**（数字形状被描亮）；下方读出「刚放电 神经元 #x（偏好数字 d）：n 个突触 +0.8，整列归一化 ×c」。`n` 与闪白像素数一致（>0）。

- [ ] **Step 5: Commit**

```bash
git add snn_demo.html
git commit -m "feat: 参数更新面板——权重网格 ΔW 闪白高亮 + 数值读出

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: 渲染循环接线与废弃函数清理

**Files:**
- Modify: `snn_demo.html:730-732`（renderStep）、`745-848`（删除 renderEnc/renderLif/renderRaster/renderVolt）、`883-888`（encBuf 清理）、`914-937`（renderStage4）、`973-981`（reset handler）

**Interfaces:**
- Consumes: Task 5 的 `renderCalc()`、Task 6 的 `decayFlash()/renderDw()`
- Produces: 完整可运行的页面

- [ ] **Step 1: renderStep 只保留必要渲染（730–732）**

旧：
```js
function renderStep() {
  renderInput(); renderEnc(); renderLif(); renderRaster(); renderVolt();
}
```
新：
```js
function renderStep() {
  renderInput(); renderCalc();
  decayFlash(); renderDw();
}
```

- [ ] **Step 2: 删除废弃渲染函数（745–848）**

删除 `renderEnc()`（745–759）、`renderLif()`（760–789）、`renderRaster()`（790–813）、`renderVolt()`（814–848）四个函数整块。`renderInput()`（733–744）保留。

- [ ] **Step 3: 清理 encBuf（不再使用）**

- 删 `const encBuf = new Float32Array(NIN);`（596）
- 删 simStep 里 `encBuf[i] = 1;`（Task 4 Step 4 区域）
- 删 resetWeights 里 `encBuf.fill(0);`（Task 4 Step 2 区域）

- [ ] **Step 4: renderStage4 各分支补 renderCalc + renderDw（914–937）**

旧：
```js
function renderStage4() {
  const main = document.getElementById("verdictMain");
  const sub = document.getElementById("verdictSub");
  if (mode === "learn") {
    main.textContent = "学习中";
    main.style.color = "";
    sub.innerHTML = trained >= TRAIN.length
      ? "已看完全部样本 —— 到「推理」模式看它认数字"
      : "正在用 STDP 学习… 观察上方权重如何自组织";
    renderProb(null);
    return;
  }
  if (trained === 0) {
    main.textContent = "?";
    main.style.color = "";
    sub.innerHTML = "还没学习。先切到 <b>学习</b> 模式播放一轮，再回来推理";
    renderProb(null);
    return;
  }
  const win = argmax(spikeAccum, N);
  main.textContent = "数字 " + pref[win];
  sub.innerHTML = "放电最多的神经元 <b>#" + win + "</b>，它偏爱数字 <b>" + pref[win] + "</b>（本图放电 " + Math.round(spikeAccum[win]) + " 次）";
  renderProb(computeProb());
}
```
新：
```js
function renderStage4() {
  const main = document.getElementById("verdictMain");
  const sub = document.getElementById("verdictSub");
  if (mode === "learn") {
    main.textContent = "学习中";
    main.style.color = "";
    sub.innerHTML = trained >= TRAIN.length
      ? "已看完全部样本 —— 到「推理」模式看它认数字"
      : "正在用 STDP 学习… 观察下方权重如何自组织";
    renderProb(null);
    renderCalc(); renderDw();
    return;
  }
  if (trained === 0) {
    main.textContent = "?";
    main.style.color = "";
    sub.innerHTML = "还没学习。先切到 <b>学习</b> 模式播放一轮，再回来推理";
    renderProb(null);
    renderCalc(); renderDw();
    return;
  }
  const win = argmax(spikeAccum, N);
  main.textContent = "数字 " + pref[win];
  sub.innerHTML = "放电最多的神经元 <b>#" + win + "</b>，它偏爱数字 <b>" + pref[win] + "</b>（本图放电 " + Math.round(spikeAccum[win]) + " 次）";
  renderProb(computeProb());
  renderCalc(); renderDw();
}
```

- [ ] **Step 5: reset handler 更新画布清空列表（973–981）**

旧：
```js
document.getElementById("btnReset").addEventListener("click", () => {
  running = false; setPlay();
  resetWeights();
  renderGrid(); renderCounts(); renderStage4();
  ["canvasIn", "canvasEnc", "canvasLif", "canvasRaster", "canvasVolt"].forEach(id => {
    const cv = document.getElementById(id);
    cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
  });
});
```
新：
```js
document.getElementById("btnReset").addEventListener("click", () => {
  running = false; setPlay();
  resetWeights();
  renderGrid(); renderCounts(); renderStage4();
  ["canvasIn", "canvasProb", "calcSpark"].forEach(id => {
    const cv = document.getElementById(id);
    if (cv) cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
  });
});
```

- [ ] **Step 6: 语法校验**

Run:
```bash
awk '/<script>/{f=1; next} /<\/script>/{f=0} f' snn_demo.html > _check.js && node --check _check.js && rm _check.js
```
Expected: 无输出、退出码 0。

- [ ] **Step 7: 浏览器回归**

打开页面：控制条、输入图、输出概率柱、演算面板、参数更新面板全部正常，无 console 报错；推理/学习/播放/暂停/重置/速度/随机/数字选择均可用。

- [ ] **Step 8: Commit**

```bash
git add snn_demo.html
git commit -m "refactor: 接线 renderStep/renderStage4/reset，删除废弃渲染函数与 encBuf

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: 全量验证（按规格 §④）

**Files:**
- Verify: `snn_demo.html`

- [ ] **Step 1: 结构检查**

打开 `snn_demo.html`：4 阶段卡片 / 栅格图 / 示波器 / 静态流程图全部消失；新两块面板出现；深色主题一致；窄窗口（<900px）io-row 纵向、无水平滚动。

- [ ] **Step 2: 推理链路检查**

切「学习」播放一轮（`trained` 到 400 / 进度条满）→ 切「推理」→ 选任意数字播放：
- 演算面板 ①②③④ 数值随图实时变化；sparkline 正常。
- 放完 ⑥ 概率柱与「数字 X」判定一致（`computeProb()` 与 `argmax` 同源）。
- ⑤ `c_d` 与 ⑥ `P` 数字自洽（`c_d/Σ = P`）。

- [ ] **Step 3: 参数更新检查**

学习播放中：网格被增强突触闪白（数字形状描亮）；ΔW 读出「n 个突触 +0.8，整列归一化 ×c」；`+0.8 = A_PLUS`、`c = W_NORM/s` 与代码一致；闪白逐帧衰减。

- [ ] **Step 4: 重置与边界**

- 重置：权重归零、演算面板回「—」/提示、闪白清零、`trained=0` 提示恢复。
- 未学习时 ⑤⑥ 显示提示；学习模式 ⑤⑥ 显示「不判定概率」。
- 全程无放电（理论边界）：⑥ 显示「本图无神经元放电」，不报错。

- [ ] **Step 5: 常量核对**

面板/读出中出现的 `0.6`（SPIKE_GAIN）、`0.94`（LEAK）、`200`（T）、`+0.8`（A_PLUS）、`78.4`（W_NORM 归一化）×c 与代码常量一致。

- [ ] **Step 6: 提交最终修复（若有）**

```bash
git add -A
git commit -m "fix: 验证中发现的问题修复

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push origin main
```

- [ ] **Step 7: 推送所有实施 commit**

```bash
git push origin main
```
Expected: 所有任务 commit（Task 1–8）推送成功。
