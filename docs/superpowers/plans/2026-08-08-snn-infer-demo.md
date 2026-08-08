# SNN 推理管线教学演示页 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一个交互式教学页 `snn_demo.html`，用预训练的单层 784→100 LIF+WTA 网络，实时演示「从 28×28 图像像素算到 0-9 分类概率」的六步计算过程（每步公式 + 真实数值）。

**Architecture:** 3 个文件。`export_snn_demo.py` 用项目真实 `snn.py` 训练网络并导出 `snn_weights.js`（权重/阈值/偏好/base64 样本图）。`snn_demo.html` 是自含教学页：顶部动画行（输入图→100 神经元脉冲→概率柱）+ 六步实时演算面板 + 权重网格 + STDP 文字说明。页面 `<script src="snn_weights.js">` 加载数据，`file://` 双击即用。

**Tech Stack:** Python 3.10 (uv/.venv)、torch 2.6、numpy；单文件 HTML + 原生 canvas JS（无框架、无外部 CDN、无 LaTeX 依赖，公式用 HTML `<sub>/<sup>` + unicode 手排）。

## Global Constraints

- 六步公式与 `snn.py` 逐行一致：`px=u/255`、`P=px×0.6`、`V=V×0.94+ΣW·spike`、WTA 取最大 V 放电、`spikeAccum++`、`pref=argmax digCount`、`P(d)=c_d/Σ`。
- JS 前向语义必须逐句复刻 `snn.py` 的 `step()`：积分 → 不应期清零 → `×leak` → WTA → fire（全部 V 清零；非不应期 refr=1；winner refr=4）→ refr 递减。
- 演示参数默认项目基线：`w_norm=16, theta_init=25, theta_clamp=(5,100), a_plus=0.8, leak=0.94, refr=4, spike_gain=0.6, T=200`；若 group-by-pref 准确率 <0.55 则改用经典 `78.4/15`。
- 页面单文件自含、深色仪器风（色值沿用 `index.html` 设计系统：`--bg0/--amber/--cyan/--green/--red/--font-mono` 等）。
- 项目无测试框架：以导出脚本自检打印 + 浏览器人工核验代替 TDD。
- 所有命令在 `d:\project\GeneralVision` 下用 `.venv\Scripts\python.exe` 运行；Windows/PowerShell。
- 每次代码改动前提交并推送（CLAUDE.md 约定）：`git add -A && git commit -m "..." && git push origin main`。

---

### Task 1: `export_snn_demo.py` —— 训练 + 验证 + 导出 `snn_weights.js`

**Files:**
- Create: `export_snn_demo.py`
- Produce: `snn_weights.js`（生成产物，需提交，供 Task 2 用）

**Interfaces:**
- Consumes: `data_loading.load_mnist()` → `(train_img[N,784] f32∈[0,1], train_lbl[N], test_img, test_lbl)`；`snn.SNN`、`snn.LayerConfig`、`snn.SNNParams`、`snn.accuracy_votes`、`snn.pref_diversity`；`genome.BASE_LAYER`。
- Produces: `snn_weights.js` 内声明 6 个全局常量（Task 2 依赖）：
  - `SNN_W_B64` — 784×100 权重列主序（`W[i*100+j]`），Float32 little-endian base64
  - `SNN_THETA_B64` — 100 个阈值，Float32 little-endian base64
  - `SNN_PREF` — 100 元素整数数组（每神经元偏好标签）
  - `SNN_IMGS_B64` — 演示图像（每数字 `N_IMGS_PER_DIGIT` 张，784 像素/张，uint8 0-255），字节流 base64
  - `SNN_IMGS_LBL` — 与图像一一对应的标签数组（0-9 各重复 N 次）
  - `SNN_IMGS_PER_DIGIT` — 整数常量

- [ ] **Step 1: 写导出脚本**

```python
"""导出 SNN 推理演示数据：预训练权重 + 神经元偏好 + 演示样本图。

用项目真实 snn（LIF+WTA+STDP）训练单层 784→100 网络，验证 group-by-pref
读出准确率后，把权重/阈值/偏好/样本图以 base64 写入 snn_weights.js，
供 snn_demo.html 用 <script src> 加载（file:// 双击可用）。

用法:
    python export_snn_demo.py            # 默认 baseline 参数，2000 样本
    python export_snn_demo.py --variant classic --train 4000
"""
from __future__ import annotations

import argparse
import base64
import os

import numpy as np
import torch

from data_loading import load_mnist
from genome import BASE_LAYER
from snn import LayerConfig, SNN, SNNParams, accuracy_votes, pref_diversity

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "snn_weights.js")
SEED = 0
N_IMGS_PER_DIGIT = 5


def _pack_f32(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def _pack_u8(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<u1").tobytes()).decode()


def _train_layer(variant: str) -> tuple[SNN, np.ndarray, np.ndarray]:
    """训练单层网络，返回 (net, tr_x, tr_y)。"""
    train_img, train_lbl, test_img, test_lbl = load_mnist()
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(train_img))
    tr_x = train_img[idx[: args.train]]
    tr_y = train_lbl[idx[: args.train]]

    if variant == "classic":
        layer = LayerConfig(n_out=100, seed=SEED, w_norm=78.4, theta_init=15.0,
                            theta_clamp=(5, 100))
    else:
        layer = LayerConfig(n_out=100, seed=SEED, **BASE_LAYER)
    params = SNNParams(spike_gain=0.6, T=200, num_classes=10,
                       input_size=784, seed=SEED)
    net = SNN([layer], params, torch.device("cpu"))

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    for i in range(len(tr_x)):                       # 一生：在线 STDP 训练
        net.train_sample(torch.tensor(tr_x[i]), int(tr_y[i]), gen)
    cal = rng.permutation(len(tr_x))[: 500]
    net.calibrate(torch.tensor(tr_x[cal]),          # 冻结权重补 dig_count
                  torch.tensor(tr_y[cal]), gen)
    return net, tr_x, tr_y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--variant", choices=["baseline", "classic"], default="baseline")
    ap.add_argument("--no-export", action="store_true")   # 只评估，不写文件
    global args
    args = ap.parse_args()

    net, tr_x, tr_y = _train_layer(args.variant)
    layer = net.layers[0]

    # 评估 group-by-pref 读出（演示页用的就是这个读出）
    train_img, train_lbl, test_img, test_lbl = load_mnist()
    va_x, va_y = test_img[:1000], test_lbl[:1000]
    gen_va = torch.Generator(device="cpu").manual_seed(SEED + 3)
    score = net.evaluate_batch(torch.tensor(va_x), gen_va)
    pref = layer.pref
    acc = accuracy_votes(score, pref, torch.tensor(va_y))
    div = pref_diversity(pref, 10)
    print(f"[{args.variant}] train={len(tr_x)} group-by-pref 准确率={acc:.3f} "
          f"pref 多样性={div}/10")
    if args.no_export:
        return

    # 演示样本图：每数字取前 N_IMGS_PER_DIGIT 张官方测试图（uint8 0-255）
    imgs, lbls = [], []
    for d in range(10):
        d_idx = np.where(test_lbl == d)[0][:N_IMGS_PER_DIGIT]
        imgs.append((test_img[d_idx] * 255).round().astype(np.uint8))
        lbls.append(np.full(len(d_idx), d, dtype=np.uint8))
    imgs = np.concatenate(imgs).reshape(-1)          # [50*784]
    lbls = np.concatenate(lbls).tolist()

    W = layer.W.detach().cpu().numpy()               # [784, 100]，列主序
    theta = layer.theta.detach().cpu().numpy()       # [100]
    js = (
        "// 由 export_snn_demo.py 生成，勿手改（重跑脚本再生）\n"
        f"const SNN_W_B64 = \"{_pack_f32(W)}\";\n"
        f"const SNN_THETA_B64 = \"{_pack_f32(theta)}\";\n"
        f"const SNN_PREF = {pref.cpu().tolist()};\n"
        f"const SNN_IMGS_B64 = \"{_pack_u8(imgs)}\";\n"
        f"const SNN_IMGS_LBL = {lbls};\n"
        f"const SNN_IMGS_PER_DIGIT = {N_IMGS_PER_DIGIT};\n"
    )
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(js)

    # 自检：base64 往返一致
    dec = np.frombuffer(base64.b64decode(_pack_f32(W)), dtype="<f4").reshape(784, 100)
    assert np.array_equal(dec, W), "权重往返不一致"
    dec_i = np.frombuffer(base64.b64decode(_pack_u8(imgs)), dtype="<u1")
    assert np.array_equal(dec_i, imgs), "图像往返不一致"
    print(f"已导出 {OUT}  "
          f"({os.path.getsize(OUT)//1024}KB)  准确率={acc:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行 baseline，检查准确率与导出**

Run: `.venv\Scripts\python.exe export_snn_demo.py`
Expected: 打印准确率、pref 多样性（期望接近 10）、`已导出 snn_weights.js (…)`；若准确率 <0.55，继续 Step 3 换 classic 参数，否则跳到 Step 4。

- [ ] **Step 3: （仅当准确率 <0.55）尝试 classic 参数**

Run: `.venv\Scripts\python.exe export_snn_demo.py --variant classic --train 4000`
Expected: 打印更高准确率；两个 variant 中选准确率高者导出（把 `--variant`/`--train` 默认值在脚本里固化，重跑一次生成最终文件）。

- [ ] **Step 4: 确认 snn_weights.js 结构**

Run: `python -c "import re; s=open('snn_weights.js',encoding='utf-8').read(); print([re.findall(r'const (\w+)', s)])"`
Expected: 输出含 `SNN_W_B64, SNN_THETA_B64, SNN_PREF, SNN_IMGS_B64, SNN_IMGS_LBL, SNN_IMGS_PER_DIGIT` 六个常量；文件大小 ≈450KB。

- [ ] **Step 5: 提交**

```bash
git add export_snn_demo.py snn_weights.js
git commit -m "feat: SNN 演示数据导出脚本与预训练权重（训练+验证+base64 导出）"
git push origin main
```

---

### Task 2: `snn_demo.html` —— 六步实时演算教学页

**Files:**
- Create: `snn_demo.html`

**Interfaces:**
- Consumes: `snn_weights.js` 的 6 个全局常量（Task 1 产物）；六步公式与 `snn.py` 语义（见 Global Constraints）。
- Produces: 可直接用浏览器打开的教学页（自含，仅依赖同目录 `snn_weights.js`）。

- [ ] **Step 1: 写页面骨架 + 设计系统 CSS**

结构自上而下（见设计文档）：

```
<header> 标题「像素 → 分类：SNN 计算过程」+ 副标题
<controls> 数字 0-9 | 🔀随机 | ▶播放/暂停 | 速度×1/×2/×5/×10 | 重置
<div class="toprow">
  canvas#in    (28×28 输入图)
  canvas#spk   (10×10=100 神经元脉冲点阵，放电时闪亮)
  canvas#prob  (0-9 概率柱)
</div>
<aside id="calcPanel"> 六行，每行：步骤号 | 公式 | 真实数值（span.live）</aside>
<canvas id="wgrid">    (10×10 神经元 × 28×28 权重块，按偏好着色边框)
<section class="stdp-note"> STDP 学习简要说明（文字+公式，无动画）</section>
```

CSS 沿用 `index.html` 设计系统变量（`--bg0/--bg1/--line1/--txt0/--amber/--cyan/--green/--red/--font-mono/--font-ui`），深色、网格纹理 `body::before`、面板 `border:1px solid var(--line1); border-radius:10px`。公式用 `font-family:var(--font-mono); color:var(--green)`，数值 `color:var(--cyan)`。

- [ ] **Step 2: 写数据解码 + JS 前向（核心，逐句对应 snn.py）**

```js
const NIN=784, N=100, T=200, LEAK=0.94, SPIKE_GAIN=0.6, REFR=4;
const IMG_PER_DIGIT = SNN_IMGS_PER_DIGIT;

function b64Bytes(b64){ const s=atob(b64); const u=new Uint8Array(s.length);
  for(let i=0;i<s.length;i++) u[i]=s.charCodeAt(i); return u; }
function b64F32(b64){ const u=b64Bytes(b64); const dv=new DataView(u.buffer);
  const a=new Float32Array(u.length/4);
  for(let i=0;i<a.length;i++) a[i]=dv.getFloat32(i*4,true); return a; }

let W=b64F32(SNN_W_B64);            // [784*100] 列主序，W[i*100+j]
let theta=b64F32(SNN_THETA_B64);    // [100]
let pref=Int32Array.from(SNN_PREF); // [100]
let imgs=b64Bytes(SNN_IMGS_B64);    // [50*784] uint8
let imgLbl=Int32Array.from(SNN_IMGS_LBL);

// ---- 每样本运行态 ----
let V=new Float32Array(N), refr=new Int32Array(N).fill(0);
let spikeAccum=new Int32Array(N), preCount=new Int32Array(NIN), t=0, done=false;

function resetSample(){
  V.fill(0); refr.fill(0); spikeAccum.fill(0); preCount.fill(0); t=0; done=false;
  fireHist.length=0; // ③ 行 fire 事件快照
}

function stepLif(spikes){
  for(let j=0;j<N;j++){            // 积分（不应期神经元积分后清零，等价不积分）
    let s=0; const off=j; for(let i=0;i<NIN;i++) s+=W[i*N+off]*spikes[i];
    V[j]+=s;
    if(refr[j]>0) V[j]=0;
    V[j]*=LEAK;
  }
  let winner=-1,bestV=-Infinity;   // WTA：refr<=0 且 V>=theta 取最大
  for(let j=0;j<N;j++){ if(refr[j]<=0 && V[j]>=theta[j] && V[j]>bestV){bestV=V[j];winner=j;} }
  if(winner>=0){
    spikeAccum[winner]++;
    const wasIdle=new Int32Array(N);
    for(let j=0;j<N;j++){ wasIdle[j]=(refr[j]<=0)?1:0; refr[j]=0; }
    for(let j=0;j<N;j++) if(wasIdle[j]) refr[j]=1;   // 非 winner 且非不应期 refr=1
    refr[winner]=REFR;                               // winner refr=4
    fireHist.push({neuron:winner,t,V:bestV,theta:theta[winner],vTrace:V.slice()});
  }
  for(let j=0;j<N;j++) if(refr[j]>0) refr[j]--;      // 不应期递减
  return winner;
}
```

注意：`_fire` 后 `V` 对**全部**神经元清零（`fireHist` 里 `vTrace` 存 fire 前快照，供迷你波形画「积分→过阈→归零」）。播放循环 `t=0..T-1`：伯努利采样 `spikes[i]=Math.random()<px[i]*SPIKE_GAIN`（并累计 `preCount`），调 `stepLif`，逐帧 `render()`。

- [ ] **Step 3: 六步演算面板 + 权重网格 + 顶部动画**

每步 `live` 值来源（随播放实时刷新，样本放完定格）：
- ① `px=u/255`：最亮像素 `px` 与 `u`、亮像素（px>0.5）个数。
- ② `P=px×0.6`：最亮像素的 `P`、该像素实际放电 `preCount/200` 次。
- ③ `V=V×0.94+ΣW·spike`：最近 fire 事件 `神经元#j @t：V=bestV ≥ θ=theta`；右侧 160×36 迷你波形画该神经元 `vTrace`（画 θ 水平线）。
- ④ `spikeAccum[j]++`：最近 fire 神经元累计 `n` 次；全体总放电 `ΣspikeAccum`。
- ⑤ `pref[j]=argmax digCount`：`c[d]=Σ_{pref[j]=d} spikeAccum[j]`，列出 `c_0..c_9`。
- ⑥ `P(d)=c_d/Σ`：概率柱 + argmax 高亮 + 判定文本「最可能是 数字 X」。
- 无放电（`ΣspikeAccum==0`）：⑥ 显示占位「本图无神经元放电」，不除零。

权重网格 `canvas#wgrid`：10×10 神经元，每个画 28×28 权重块（`W[i*100+j]` 灰度，白=正大值）；神经元边框色 = `DCOL[pref[j]]`（10 色调色板）；右下角标偏好数字。

- [ ] **Step 4: 浏览器核验（run skill 打开）**

- [ ] 打开 `snn_demo.html`：六步面板 + 动画行 + 权重网格 + STDP 说明都在，无水平滚动，配色与 index.html 一致。
- [ ] 点几个数字播放：①②③④ 数值随步进实时变；样本放完 ⑤⑥ 概率柱与判定文本出现且与 argmax 一致。
- [ ] 抽查与 Python 一致性：同一数字（如「3」）在浏览器里多次播放，判定大多数为「3」（泊松随机导致偶发翻转可接受）；权重网格能看到数字形权重。
- [ ] 随机/重置/速度切换均正常；多次播放无 NaN、无报错。
- [ ] 窗口 <900px：面板不溢出。

- [ ] **Step 5: 提交**

```bash
git add snn_demo.html
git commit -m "feat: SNN 推理管线教学演示页——像素→分类六步实时演算"
git push origin main
```

---

## 自检记录

- **Spec coverage:** 六步公式（Task 2 Step 3）✓；动画行（Step 1 toprow + Step 3）✓；权重网格（Step 3）✓；STDP 文字说明（Step 1 section）✓；预训练权重导出（Task 1）✓；边界/除零（Step 3）✓；窄窗口（Step 4）✓。
- **Placeholder scan:** 无 TBD/TODO；所有代码步骤含完整代码。
- **Type consistency:** `SNN_W_B64/SNN_THETA_B64/SNN_PREF/SNN_IMGS_B64/SNN_IMGS_LBL/SNN_IMGS_PER_DIGIT` 六常量在 Task 1 产出与 Task 2 消费处完全同名；JS 内 `W[i*N+j]` 列主序与导出脚本 `W.detach().numpy()[784,100]` 一致。
