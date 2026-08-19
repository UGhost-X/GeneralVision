# GeneralVision — ROI 标注 / 处理工作台

本地 Web 应用（Python 标准库 http.server + OpenCV/numpy + 可选 PyTorch）：

1. **标注台**：上传图像、画矩形 ROI、裁剪后放大查看（等比缩放）
2. **ROI 处理台**：多选 ROI 批量处理
   - 传统算法（numpy/OpenCV 实现）：维纳滤波、盲解卷积 (RL)、一阶/二阶导数极值法、Zernike 矩亚像素边缘提取
   - 深度学习清晰化（可选，需安装 torch + 权重）：
     - **HAT**：真实世界 4x 超分/恢复（`Real_HAT_GAN_sharper.pth` 更锐利 / `HAT_SRx4_ImageNet-pretrain.pth` 更保真），支持 4x 超分输出或恢复后同尺寸输出
     - **Restormer**：运动去模糊、散焦去模糊、真实去噪、高斯灰度去噪 σ25（输出与原图同尺寸）

## 启动

```powershell
.venv\Scripts\python.exe roi_workbench.py --host 127.0.0.1 --port 8768
```

浏览器打开 http://127.0.0.1:8768

## 安装深度学习模型（HAT / Restormer）

基础依赖只有 numpy + opencv。需要 HAT/Restormer 时：

```powershell
# 1) 安装 PyTorch + einops（优先 GPU 版，无 NVIDIA GPU 再装 CPU 版）
#    GPU 版（CUDA 12.4）：
uv pip install --python .venv\Scripts\python.exe torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
#    CPU 版：
#    uv pip install --python .venv\Scripts\python.exe torch einops --index-url https://download.pytorch.org/whl/cpu
#    若官方源慢/不稳：scripts/download_models.py 会把 CPU wheel 下载到
#    %TEMP%\roi_models\torch_cpu.whl，然后执行：
#    uv pip install --python .venv\Scripts\python.exe %TEMP%\roi_models\torch_cpu.whl einops

# 2) 下载模型权重（HAT 2 个 + Restormer 4 个，约 660MB，支持断点续传，多源自动回退）
.venv\Scripts\python.exe scripts/download_models.py
```

权重保存在 `models/`（已 gitignore）。处理台页面会显示深度学习模型就绪状态与推理设备（GPU/CPU）；
torch 或权重缺失时，运行对应算法会返回明确提示，不影响传统算法使用。
推理会自动检测 CUDA：有可用 GPU 时走 GPU（RTX 3060 上 HAT 4x 约 0.7s/64px、Restormer 约 0.2s/128px），否则回退 CPU。

## 说明

- 传统去模糊算法（维纳/RL）假设图像确实被给定 PSF 模糊；若原图本来清晰或 PSF 不匹配，
  会出现振铃/变糊，可调大 NSR、降低恢复强度、减少迭代，或改用边缘类算法。
- HAT/Restormer 为学习式模型，对未见过的退化类型效果有限；若结果过锐/过度平滑，
  可调大「与原图混合强度」保留原始信息。
