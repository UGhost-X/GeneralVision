# SNN 演示页「像素→概率」计算流程图 + 0-9 概率输出 · 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 snn_demo.html 里令人困惑的「七步学习链路详解」替换为一张「像素→概率」6 节点计算流程图，并让推理输出真实的 0-9 概率柱状图。

**Architecture:** 纯单文件 HTML/CSS/JS 就地修改。删除旧 walkthrough 区块（CSS + HTML），插入内联 SVG 计算流程图（静态、无 JS）；新增 `computeProb()`（按偏好数字分组、归一化放电计数）与 `renderProb()`（canvas 画 0-9 水平概率柱），并在 `renderStage4()` 推理分支调用。核心逻辑 `simStep()`/`fire()`/`resetWeights()` 一字不动。

**Tech Stack:** 原生 HTML5 + CSS 自定义属性（`var(--panel)` 等令牌）+ SVG + Canvas 2D。无构建、无测试框架（仓库现状）。

## Global Constraints

- 只改 `d:\project\GeneralVision\snn_demo.html` 一个文件。
- 不得改动 `simStep()` / `fire()` / `resetWeights()` / `onSampleDone()` 的核心逻辑。
- 颜色一律用现有 CSS 变量；canvas 绘制中硬编码的十六进制值必须与 CSS 变量值一致（amber `#FFB454`、cyan `#4CC9F0`、ink-dim `#8B94A8`）。
- 仓库无测试运行器（无 pyproject/setup/test 文件），每个任务的"测试"是浏览器人工核对（打开页面 → 按步骤操作 → 看指定结果）。
- Git 工作流（CLAUDE.md）：每个任务完成后 `git add <具体文件>` → commit → `git push origin main`；若 push 因远端领先被拒，先 `git pull --rebase origin main` 再 push。**不要** `git add -A`（会把用户未跟踪的实验文件 `snn.py`、`data_loading.py`、`_*.py` 卷进提交）。

## File Structure

唯一改动文件 `snn_demo.html`，三处区域：

| 区域 | 当前大致行号 | 本计划改动 |
|---|---|---|
| `<style>` | 293–311（walk 七步 CSS） | 替换为 flow 流程图 CSS |
| HTML 主体 learn-panel 之后 | 470（learn-cap 文字）、477–578（walk 区块）、428–429（Stage 4 verdict） | learn-cap 改指流程图；walk 区块替换为 flow 区块；Stage 4 加概率 canvas |
| `<script>` | 923–943（renderStage4） | 新增 computeProb/renderProb，renderStage4 各分支接入 |

---

### Task 1: 替换「七步详解」为「像素→概率」计算流程图

**Files:**
- Modify: `snn_demo.html`（`<style>` 内 walk CSS → flow CSS；HTML 内 walk 区块 → flow 区块）

**Interfaces:**
- Produces: `.flow-panel` / `.flow-title` / `.flow-intro` / `.flow-svg` / `.flow-node-num` / `.flow-node-title` / `.flow-node-math` / `.flow-arrow` 这些类名，供页面渲染；Task 3 不依赖它们。

- [ ] **Step 1: 替换 CSS（删除 walk 样式，插入 flow 样式）**

用 Edit 工具，old_string 为 `<style>` 内从 `  /* ---------- step walkthrough ---------- */` 起到 `.wstep-body code { ... }` 规则结束（含其后的空行、`@media` 之前）为止的整块（即 293 行到 311 行之间的内容）。new_string 替换为：

```css
  /* ---------- pixel→probability flow ---------- */
  .flow-panel {
    margin: 26px 0 0;
    padding: 20px 20px 18px;
    background: linear-gradient(180deg, #121A2B, var(--panel));
    border: 1px solid var(--rule);
    border-radius: 8px;
  }
  .flow-title { font-family: var(--mono); font-size: 17px; font-weight: 600; margin: 0 0 6px; }
  .flow-intro { color: var(--ink-dim); font-size: 13px; margin: 0 0 14px; max-width: 70ch; }
  .flow-svg { display: block; width: 100%; max-width: 1120px; height: auto; margin: 0 auto; }
  .flow-node-num { font-family: var(--mono); font-size: 13px; fill: var(--amber); }
  .flow-node-title { font-family: var(--mono); font-size: 14px; font-weight: 600; fill: var(--ink); }
  .flow-node-math { font-family: var(--mono); font-size: 13px; fill: var(--green); }
  .flow-arrow { stroke: var(--ink-faint); fill: var(--ink-faint); stroke-width: 2; }

```

（末尾保留原有空行与 `@media`。）

- [ ] **Step 2: 替换 HTML 区块（walk → flow）**

先改区块开标签：`<section class="walk-panel" aria-label="学习链路七步分步讲解">` → `<section class="flow-panel" aria-label="从图片像素到输出分类概率的计算流程">`。

再用 Edit 把该 section 内部（`<p class="walk-title">` 到 `</ol>` 之间的全部内容，即原来标题+引子+7 个 li）替换为新的标题+引子+SVG。`</section>` 闭合标签保留不动。new_string：

```html
    <p class="flow-title">像素 → 概率：这张图是怎么算出来的</p>
    <p class="flow-intro">
      从一张 28×28 灰度图，到 0–9 每个数字的概率：这 6 步就是一次完整的前向计算。
      每步下方标出它用的公式与常量。
    </p>
    <svg class="flow-svg" viewBox="0 0 1120 200" role="img" aria-label="从像素到概率的六步计算流程">
      <g transform="translate(8,25)">
        <rect width="165" height="150" rx="10" fill="var(--panel-2)" stroke="var(--rule)"/>
        <text x="82" y="32" text-anchor="middle" class="flow-node-num">① 输入像素</text>
        <text x="82" y="62" text-anchor="middle" class="flow-node-title">28×28 灰度图</text>
        <text x="82" y="88" text-anchor="middle" class="flow-node-title">784 维向量</text>
        <text x="82" y="124" text-anchor="middle" class="flow-node-math">px[i] ∈ [0,1]</text>
      </g>
      <line class="flow-arrow" x1="176" y1="100" x2="191" y2="100"/>
      <polygon class="flow-arrow" points="195,100 186,95 186,105"/>
      <g transform="translate(195,25)">
        <rect width="165" height="150" rx="10" fill="var(--panel-2)" stroke="var(--rule)"/>
        <text x="82" y="32" text-anchor="middle" class="flow-node-num">② 速率编码</text>
        <text x="82" y="62" text-anchor="middle" class="flow-node-title">越亮越发脉冲</text>
        <text x="82" y="88" text-anchor="middle" class="flow-node-title">泊松采样</text>
        <text x="82" y="124" text-anchor="middle" class="flow-node-math">P = px × 0.6</text>
      </g>
      <line class="flow-arrow" x1="363" y1="100" x2="378" y2="100"/>
      <polygon class="flow-arrow" points="382,100 373,95 373,105"/>
      <g transform="translate(382,25)">
        <rect width="165" height="150" rx="10" fill="var(--panel-2)" stroke="var(--rule)"/>
        <text x="82" y="32" text-anchor="middle" class="flow-node-num">③ 积分放电</text>
        <text x="82" y="62" text-anchor="middle" class="flow-node-title">加权积分 + 漏电</text>
        <text x="82" y="88" text-anchor="middle" class="flow-node-title">阈值 + 胜者全取</text>
        <text x="82" y="124" text-anchor="middle" class="flow-node-math">V += W·spike</text>
      </g>
      <line class="flow-arrow" x1="550" y1="100" x2="565" y2="100"/>
      <polygon class="flow-arrow" points="569,100 560,95 560,105"/>
      <g transform="translate(569,25)">
        <rect width="165" height="150" rx="10" fill="var(--panel-2)" stroke="var(--rule)"/>
        <text x="82" y="32" text-anchor="middle" class="flow-node-num">④ 放电计数</text>
        <text x="82" y="62" text-anchor="middle" class="flow-node-title">T = 200 步</text>
        <text x="82" y="88" text-anchor="middle" class="flow-node-title">每神经元计数</text>
        <text x="82" y="124" text-anchor="middle" class="flow-node-math">spikeAccum[j]</text>
      </g>
      <line class="flow-arrow" x1="737" y1="100" x2="752" y2="100"/>
      <polygon class="flow-arrow" points="756,100 747,95 747,105"/>
      <g transform="translate(756,25)">
        <rect width="165" height="150" rx="10" fill="var(--panel-2)" stroke="var(--rule)"/>
        <text x="82" y="32" text-anchor="middle" class="flow-node-num">⑤ 按数字分组</text>
        <text x="82" y="62" text-anchor="middle" class="flow-node-title">神经元 → 偏好</text>
        <text x="82" y="88" text-anchor="middle" class="flow-node-title">100 → 10 组</text>
        <text x="82" y="124" text-anchor="middle" class="flow-node-math">pref[j]</text>
      </g>
      <line class="flow-arrow" x1="924" y1="100" x2="939" y2="100"/>
      <polygon class="flow-arrow" points="943,100 934,95 934,105"/>
      <g transform="translate(943,25)">
        <rect width="165" height="150" rx="10" fill="var(--panel-2)" stroke="var(--rule)"/>
        <text x="82" y="32" text-anchor="middle" class="flow-node-num">⑥ 概率分布</text>
        <text x="82" y="62" text-anchor="middle" class="flow-node-title">0–9 概率条</text>
        <text x="82" y="88" text-anchor="middle" class="flow-node-title">归一化成概率</text>
        <text x="82" y="124" text-anchor="middle" class="flow-node-math">P(d) = c_d / Σ</text>
      </g>
    </svg>
```

- [ ] **Step 3: 浏览器验证**

打开 `snn_demo.html`（浏览器双击或 `start snn_demo.html`）。滚动到 learn-panel 下方：旧「七步详解」消失，出现标题「像素 → 概率：这张图是怎么算出来的」+ 6 个横排节点（①输入像素…⑥概率分布）、每节点 3 行文字（琥珀号/白标题/绿公式）、节点间箭头正常。窄窗口（<900px）下 SVG 等比缩小、不溢出。控制台无报错。

- [ ] **Step 4: 提交**

```bash
git add snn_demo.html
git commit -m "feat: 用「像素→概率」计算流程图替换七步详解区块"
git push origin main
```

---

### Task 2: 更新 learn-panel 引导文字

**Files:**
- Modify: `snn_demo.html`（`.learn-cap` 段落，约 470 行）

**Interfaces:**
- Consumes: Task 1 的 flow 区块标题（「像素 → 概率」）。
- Produces: 无新接口；只改文案。

- [ ] **Step 1: 改引导语**

old_string（约 470 行）：

```
        这 100 个神经元从<b>随机噪声</b>出发，仅靠放电时序自组织出偏好——
        颜色 = 它最常为哪个数字放电。完整机制见下方<b>七步详解</b>。
```

new_string：

```
        这 100 个神经元从<b>随机噪声</b>出发，仅靠放电时序自组织出偏好——
        颜色 = 它最常为哪个数字放电。下方是<b>像素 → 概率</b>的计算流程图。
```

- [ ] **Step 2: 浏览器验证**

刷新页面：learn-panel 内文字变成"……下方是 像素 → 概率 的计算流程图"，无残留"七步详解"字样。

- [ ] **Step 3: 提交**

```bash
git add snn_demo.html
git commit -m "docs: learn-panel 引导语指向像素→概率流程图"
git push origin main
```

---

### Task 3: 推理输出改为 0-9 概率柱状图（canvas）

**Files:**
- Modify: `snn_demo.html`（Stage 4 HTML 428–429 附近；`<script>` 923–943 附近）

**Interfaces:**
- Consumes: 全局数组 `pref`（每神经元偏好数字 0–9）、`spikeAccum`（本样本每神经元放电计数）、`N = 100`。
- Produces:
  - `function computeProb(): number[] | null` — 返回 10 元素概率数组（和=1），`total===0` 时返回 `null`。
  - `function renderProb(prob: number[] | null): void` — 画 `#canvasProb`；`prob` 为 null 时清空画布。
  - `renderStage4()` 三个分支末尾均调用 `renderProb(...)`。

- [ ] **Step 1: Stage 4 HTML 插入概率 canvas**

old_string：

```
      </div>
      <p class="stage-cap">整张图看完后，<b>放电最多的那个神经元</b>就是网络的"答案"——它专门为某个数字放电。</p>
```

new_string：

```
      </div>
      <div class="canvas-host">
        <canvas id="canvasProb" width="360" height="110" class="line" role="img"
                aria-label="0到9数字的分类概率柱状图"></canvas>
      </div>
      <p class="stage-cap">整张图看完后，<b>放电最多的那个神经元</b>就是网络的"答案"；下方是 0–9 每个数字的放电占比（概率）。</p>
```

- [ ] **Step 2: 在 `<script>` 中新增 `computeProb` 与 `renderProb`（插在 `function renderStage4() {` 之前）**

```js
// ---------- probability output ----------
function computeProb() {
  const counts = new Array(10).fill(0);
  let total = 0;
  for (let j = 0; j < N; j++) { counts[pref[j]] += spikeAccum[j]; total += spikeAccum[j]; }
  return total > 0 ? counts.map(c => c / total) : null;
}

function renderProb(prob) {
  const cv = document.getElementById("canvasProb");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  if (!prob) return;
  const padL = 24, padR = 6, rowH = H / 10;
  let maxD = 0;
  for (let d = 1; d < 10; d++) if (prob[d] > prob[maxD]) maxD = d;
  ctx.font = "11px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let d = 0; d < 10; d++) {
    const y = d * rowH;
    ctx.fillStyle = "#8B94A8";
    ctx.fillText(String(d), padL - 8, y + rowH * 0.5);
    const bw = Math.max(2, prob[d] * (W - padL - padR));
    ctx.fillStyle = d === maxD ? "#FFB454" : "#4CC9F0";
    ctx.globalAlpha = d === maxD ? 1 : 0.5;
    ctx.fillRect(padL, y + rowH * 0.16, bw, rowH * 0.68);
    ctx.globalAlpha = 1;
  }
}
```

- [ ] **Step 3: 修改 `renderStage4()`，三个分支末尾各加一行 `renderProb(...)`**

逐处插入（old → new）：

分支 1（learn 模式，`sub.innerHTML = ... : "正在用 STDP 学习… 观察上方权重如何自组织";` 之后的 `return;` 前）：
```js
    sub.innerHTML = trained >= TRAIN.length
      ? "已看完全部样本 —— 到「推理」模式看它认数字"
      : "正在用 STDP 学习… 观察上方权重如何自组织";
    renderProb(null);
    return;
```

分支 2（trained===0）：
```js
    sub.innerHTML = "还没学习。先切到 <b>学习</b> 模式播放一轮，再回来推理";
    renderProb(null);
    return;
```

分支 3（推理判定，`sub.innerHTML = "放电最多的神经元 ... 次）";` 之后）：
```js
  sub.innerHTML = "放电最多的神经元 <b>#" + win + "</b>，它偏爱数字 <b>" + pref[win] + "</b>（本图放电 " + Math.round(spikeAccum[win]) + " 次）";
  renderProb(computeProb());
```

- [ ] **Step 4: 浏览器验证**

打开页面 → 切「学习」→ ▶ 播放，等 `已学习 400/400`（或先点「重置权重」后重学）→ 切「推理」→ 随机/选个数字 → ▶ 播放：Stage 4 在「数字 X」下方出现 10 根水平概率柱（0–9 标签），与当前数字吻合的柱最长且为琥珀色，其余青色半透明。反复播放不同数字，概率条随结果变化。未学习时概率区为空（不显示旧图）。控制台无报错。

- [ ] **Step 5: 提交**

```bash
git add snn_demo.html
git commit -m "feat: 推理输出 0-9 概率柱状图（computeProb + renderProb）"
git push origin main
```

---

## Self-Review 结果

- **Spec 覆盖**：设计文档 7 项改动全被映射——①/②/③ 流程图替换 → Task 1；④ learn-cap → Task 2；⑤ Stage 4 canvas + ⑥ computeProb/renderProb + ⑦ 概率柱样式（canvas 内联绘制，无需额外 CSS）→ Task 3。✓
- **占位符**：无 TBD/「待补」；每步含可执行的精确 old/new 代码。✓
- **类型一致**：`computeProb()` 返回 `number[] | null`，`renderProb(prob)` 与 `renderStage4()` 三处调用类型匹配；引用变量 `pref`/`spikeAccum`/`N` 均为现有全局（已核实）。✓
