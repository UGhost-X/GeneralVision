# SNN 演示页：像素→概率 计算流程图 + 0-9 概率输出

日期：2026-08-08
文件：`snn_demo.html`（单文件自含，唯一改动目标）

## 背景

用户查看 `snn_demo.html`（LIF + 胜者全取 + STDP 的单层脉冲网络教学演示）后，对现有「七步学习链路详解」文字区块（`231f458` 提交引入）感到困惑。真正需要的是**从图片像素到输出分类概率的计算过程图示**，而不是识别/学习过程的可视化讲解。

已与用户确认的三点：
1. **demo 真实输出 0-9 概率分布**（推理结束后按数字分组的放电计数归一化成概率）。
2. 用**静态 SVG 计算流程图**（6 个节点）替换现有「七步详解」区块；概率柱状图用 **canvas** 绘制。
3. **最小改动**：只动七步详解区块与推理输出，其余可视化（4 张 stage 卡片、栅格图/示波器、权重网格学习面板）全部保留。

## 非目标

- 不做动画、不可视化 STDP 学习过程。
- 不改 `simStep()` / `fire()` / `resetWeights()` 核心逻辑。
- 不动 `.scope-grid`、`.learn-panel` 的现有结构。

## 改动点（仅 `snn_demo.html` 一个文件）

| # | 改动 | 位置（当前行号） |
|---|---|---|
| 1 | 删除七步详解 CSS 块 `.walk-panel`/`.wstep*` | 293–311 |
| 2 | 删除七步详解 HTML 块 `<!-- step-by-step walkthrough -->` | 477–578 |
| 3 | 同位置插入计算流程图区块（`.flow-panel`，内联 SVG） | 477 处 |
| 4 | `.learn-cap` 文字「完整机制见下方七步详解」→ 指向计算流程图 | 490 附近 |
| 5 | Stage 4 HTML 增加概率柱状图 canvas `#canvasProb` | 429 之后 |
| 6 | 新增 JS `computeProb()` + `renderProb()`；`renderStage4()` 末尾调用 | 940 附近 |
| 7 | 新增 CSS：`.flow-panel`、SVG 节点样式、概率柱样式 | 291 之后 |

## ① 计算流程图（内联 SVG，6 节点）

横向 6 节点、箭头连接，每节点 3 行文本：**琥珀色节点号 / mono 标题 / 绿色公式**。整体深色面板风格（`var(--panel)`、`var(--rule)`、`var(--amber)`、`var(--cyan)`、`var(--green)`），不新增色值。

```
① 输入像素       ② 速率编码        ③ 积分放电           ④ 放电计数      ⑤ 按数字分组      ⑥ 概率分布
28×28 灰度图    越亮越发脉冲     加权积分+漏电         T=200 步        神经元→偏好       0-9 概率条
px[i]∈[0,1]     P=px×0.6        V+=W·spike           spikeAccum[j]    pref[j]           P(d)=count_d/Σ
784 维向量      泊松采样         V×=0.94 · V≥θ · 胜者全取  每神经元+1      100→10 组        归一化成概率
```

节点公式与代码常量一一对应：`SPIKE_GAIN = 0.6`、`LEAK = 0.94`、`T = 200`、`pref[j]`。

- 实现：单个 `<svg viewBox="0 0 1120 200">`，`width:100%; max-width:1120px`，节点用 `<rect>` + `<text>`（`<tspan>` 分 3 行），箭头用 `<line>` + `<polygon>`。
- 响应式：viewBox 等比缩放，900px 以下随容器缩小；文本短，可读性可接受。
- 区块标题：「像素 → 概率：这张图是怎么算出来的」；一句话引导语（不带 STDP/学习内容）。

## ② 概率计算（`computeProb()`）

推理样本跑满 `T = 200` 步后调用。逻辑：

```js
function computeProb() {
  const counts = new Array(10).fill(0);
  let total = 0;
  for (let j = 0; j < N; j++) { counts[pref[j]] += spikeAccum[j]; total += spikeAccum[j]; }
  return total > 0 ? counts.map(c => c / total) : null;
}
```

- 按每个神经元的偏好数字 `pref[j]`（学习时由 `updatePref()` 从 `digCount` 取 argmax 得到）把放电次数 `spikeAccum[j]` 分进 0–9 十组，归一化成概率。
- 未放电的神经元贡献 0，天然被忽略。
- 返回 `null`（`total === 0`）时调用方显示占位。

## ③ 概率渲染（`renderProb()`，canvas）

- 新增 canvas `#canvasProb`（置于 Stage 4 verdict 之下，复用 `.line` 画布样式）。
- 画 10 根水平柱：左列标签 `0..9`，柱宽 ∝ `prob[d]`，最大值柱用琥珀色高亮、其余青色/暗色。
- 顶部保留现有 `verdictMain`「数字 X」判定（即原有 `argmax(spikeAccum)` → `pref[win]`），概率条作为其量化补充。

## ④ 边界情况

- `trained === 0`（未学习）：保持现有「还没学习」提示，不调用概率渲染。
- `computeProb()` 返回 `null`（全程无放电）：画平直占位柱或不绘制，避免除以零。
- 学习模式下 `renderStage4()` 早退路径不变（显示「学习中」）。

## ⑤ 验证方式

1. 浏览器打开 `snn_demo.html`：七步详解消失，learn-panel 下方出现 6 节点计算流程图，公式/箭头正确，深色主题一致。
2. 切「学习」播放一轮（`trained` 到 400）后切「推理」，输入任意数字播放：Stage 4 出现 0-9 概率柱状图，柱高与该数字吻合；「最可能是 数字 X」仍显示。
3. 常量核对：流程图与代码中 `SPIKE_GAIN=0.6`、`LEAK=0.94`、`T=200`、`pref[j]` 一致。
4. 回归：栅格图、示波器、权重网格、play/reset 均正常（核心 JS 未动）。
5. 窄窗口（<900px）：流程图缩放不溢出，无水平滚动。
