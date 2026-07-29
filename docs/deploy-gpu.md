# GPU 部署指南（paddlepaddle-gpu）

本服务默认安装 **CPU 版** paddlepaddle（`pip install -r requirements.txt`），任何机器都能跑。
有 NVIDIA GPU 的机器（RTX 30/40/50 系列、CUDA 12.x）切到 **GPU 版** 后，PP-OCRv6 的识别阶段
（Transformer）吞吐可提升 **5–10 倍**，是性价比最高的单点优化。

> 本文以 **Windows + RTX 4060** 为例，Linux 流程一致（把 `.venv/Scripts/` 换成 `.venv/bin/`）。

---

## 1. 前置检查

先确认 GPU 与驱动：

```bash
nvidia-smi
```

需要看到：
- `CUDA Version: 12.x`（驱动支持的最高 CUDA 版本，向后兼容 paddle 的 cu12x 轮子）
- GPU 显存 ≥ 4GB（PP-OCRv6 medium 模型加载约 0.6–1GB，推理峰值含 batch 通常 < 3GB，8GB 卡绰绰有余）

确认当前装的是 CPU 版（应输出 `False 0`）：

```bash
.venv/Scripts/python.exe -c "import paddle; print(paddle.is_compiled_with_cuda(), paddle.device.cuda.device_count())"
```

---

## 2. 换装 GPU 版 paddlepaddle

```bash
# 1) 卸载 CPU 版（paddlepaddle 与 paddlepaddle-gpu 不能共存）
.venv/Scripts/pip.exe uninstall -y paddlepaddle

# 2) 安装 GPU 版（CUDA 12.6，已在本机 RTX 4060 验证通过）
.venv/Scripts/pip.exe install paddlepaddle-gpu==3.3.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

> ⚠️ **关键坑**：PyPI 上 `pip install paddlepaddle-gpu` 默认只装到 2.6.x（旧版，
> 不支持 PaddleOCR 3.7.0 需要的 Paddle 3.x）。Paddle 3.x 的 GPU 轮子**不在 PyPI**，
> 必须用上面的 `-i https://www.paddlepaddle.org.cn/packages/stable/cu126/` 官方索引。
>
> ⚠️ **不要装 3.0.0**：3.0.0 在加载 PP-OCRv6 det 模型时会报
> `Type of attribute: strides is not right`（PIR 算子属性 bug）。用 **3.3.1** 或更高。
>
> cu126 索引上可选版本：3.0.0 / 3.1.0 / 3.2.0 / 3.2.1 / 3.2.2 / 3.3.0 / 3.3.1。
> 选与 CPU 端 paddlepaddle 对齐的最新一档。驱动向后兼容，例如驱动 `nvidia-smi`
> 显示 CUDA 13.x 仍可使用 cu126 轮子。

### 2a. 对齐 cuDNN 版本（重要）

paddlepaddle-gpu 3.3.1 的**二进制**是用 **cuDNN 9.9** 编译的（运行时会打印
`compiled with CUDNN 9.9`），但安装时 pip 默认拉的 `nvidia-cudnn-cu12` 是 9.5.1，
两者不一致会在推理时打警告：

```
The installed Paddle is compiled with CUDNN 9.9, but CUDNN version in your machine is 9.5,
which may cause serious incompatible bug.
```

装匹配二进制的版本（国内建议走清华镜像，767MB 从默认源拉很慢）：

```bash
.venv/Scripts/pip.exe install nvidia-cudnn-cu12==9.9.0.52 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> ⚠️ **已知的 pip 元数据矛盾**：paddlepaddle-gpu 3.3.1 的 pip 依赖声明 pin 的是
> `nvidia-cudnn-cu12==9.5.1.17`，但运行时又说自己是"compiled with CUDNN 9.9"——
> 这是 paddle 包自身的元数据 bug（声明与二进制自相矛盾）。装 9.9 会触发 pip 报
> `incompatible`，但**实际运行反而更正确**（匹配二进制真实编译版本，警告消失，
> probe 验证通过）。保持 9.9 即可；若哪天 `pip check` 报错扰人，可忽略或加
> `--no-deps` 重装 paddle 本身。装回 9.5 也能跑（probe 通过），只是带运行时警告。

---

## 3. 验证安装

```bash
.venv/Scripts/python.exe -c "import paddle; print(paddle.is_compiled_with_cuda(), paddle.device.cuda.device_count())"
```

期望输出：

```
True 1
```

- `True` —— paddle 是 CUDA 编译版
- `1` —— 检测到 1 块 GPU（多卡会是 2/3/…）

若仍为 `False 0`：检查上一步的轮子是否装错、是否还有残留的 CPU 包。

### 跑 smoke test

```bash
.venv/Scripts/python.exe scripts/probe_paddle.py
```

期望退出码 `0`（合成图片能正常跑通 PP-OCRv6 的 det+rec）。首次启用 FP16 时，
第一次推理会比平时慢十几秒——这是 TensorRT 在编译推理引擎并缓存，属正常现象。

---

## 4. 配置 `.env`

```ini
# 推理设备：auto = 有 GPU 用 GPU、无 GPU 回退 CPU（一份 .env 两种机器通用）
#           gpu  = 强制 GPU（无 GPU 会报错）
#           cpu  = 强制 CPU
OCR_DEVICE=auto

# FP16 半精度，仅在 OCR_DEVICE=gpu 下生效（走 TensorRT）。
# ⚠️ 见下方 §2b —— 当前 paddlepaddle-gpu 3.3.1 + TRT 11.x 组合有 native 崩溃，
#    建议 OCR_USE_FP16=false（纯 GPU FP32 已比 CPU 快一个数量级）。TRT 解决后再开。
OCR_USE_FP16=false
```

> 想拿到 FP16 收益，**必须显式设 `OCR_DEVICE=gpu`**——`auto` 下即使检测到 GPU
> 也不会自动开 FP16（避免在你没准备好的情况下意外触发 TensorRT 编译）。

### 2b. ⚠️ 已知坑：FP16/TRT 当前不可用（paddlepaddle-gpu 3.3.1 + TRT 11.x）

`OCR_USE_FP16=true` 会让 detector 传 `precision="fp16" + use_tensorrt=True`，触发
TensorRT 引擎编译。但本机实测组合下会 **native 崩溃**（进程在加载 PP-LCNet
textline 方向模型、编译其 TRT 引擎时崩，exit code 2816/0xB00，无 Python traceback）。

**已排除的尝试**：
- ✅ 已装 `tensorrt-cu12==11.1.0.106`（含 `nvinfer_11.dll` 等 C++ 库）
- ✅ 已建无版本号别名（`nvinfer.dll` / `nvinfer_plugin.dll` / `nvonnxparser.dll`
  → 指向 `_11` 版本），解决了 Paddle dynload "找无版本号 nvinfer.dll" 的第一层报错
- ❌ 仍 native 崩溃 —— 判定为 **Paddle 3.3.1 编译时绑定的 TRT 版本与 pip 上
  tensorrt-cu12 11.x 不匹配**（Paddle 的 TRT 集成对版本要求严格）

**当前结论**：保持 `OCR_USE_FP16=false`，用纯 GPU **FP32**。FP32 已比 CPU 快一个
数量级（probe：0.5s/次，CPU 时代真实任务动辄几十~几百秒）。FP16 的边际收益不值得
现在深挖 TRT 版本对齐。

**若要继续追 FP16**（未来工作）：
1. 查 Paddle 3.3.1 编译时绑定的 TRT 具体版本（paddle.version API 在 GPU 包上不可靠）
2. 从 NVIDIA 官网下对应版本的 TensorRT SDK（不是 pip），配 `PADDLE_TENSORRT_LIB_PATH`
3. 或等 paddlepaddle-gpu 发布与 TRT 11.x 对齐的新轮子
4. 服务 PATH 需加 `tensorrt_libs` 目录（见 §6 注意事项），否则 dynload 找不到 dll

启动服务后看 `/logs`，ready 日志应出现：

```
PaddleOCR ready: ... device=gpu ... fp16=True
```

---

## 5. 回退 CPU

两种方式：

**A. 不卸载 GPU 包，仅切设备**（最快，便于排查）：

```ini
OCR_DEVICE=cpu
OCR_USE_FP16=false
```

**B. 彻底换回 CPU 包**：

```bash
.venv/Scripts/pip.exe uninstall -y paddlepaddle-gpu
.venv/Scripts/pip.exe install paddlepaddle>=3.0
```

---

## 6. 注意事项

- **显存**：8GB 卡（如 RTX 4060）够用。若同时跑其它 GPU 进程或 `OCR_REC_BATCH_SIZE`
  调得很大，可能 OOM——降 batch 或关 FP16（FP16+TensorRT 会预分配引擎缓存显存）。
- **首次推理慢**：开启 FP16 后第一次推理会编译 TensorRT 引擎（十几秒～几十秒），
  缓存后后续推理即享 FP16 加速。
- **并发**：服务对 `PaddleOCR.predict()` 做了串行化（`_predict_lock`），GPU 路径同样
  串行——不会因并发把显存打爆。如需更高并发，考虑多实例 + 上层负载均衡。
- **服务部署**：Windows 生产环境仍走 NSSM（见 `docs/OPERATIONS.md`），GPU 包替换后
  重启服务即可，无需改 NSSM 配置。
