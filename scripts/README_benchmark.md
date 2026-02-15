# 模型性能基准测试使用指南

## 概述

`scripts/benchmark_model.py` 用于公平对比不同模型的部署成本，包括参数量、计算量、推理延迟、吞吐量和显存占用。

## 核心特性

### 1. 模型复杂度统计
- **参数量 (Params)**: 模型总参数数
- **GFLOPs**: 浮点运算量（使用模型自带info方法）

### 2. 精确的GPU延迟测量
- 使用 `torch.cuda.Event` 精确测量GPU延迟
- 使用 `torch.cuda.synchronize()` 避免异步误差
- CPU模式使用 `time.perf_counter()`

### 3. 延迟统计
- `latency_mean_ms`: 平均延迟
- `latency_p50_ms`: 中位数延迟
- `latency_p95_ms`: 95百分位延迟
- `latency_p99_ms`: 99百分位延迟
- `fps_mean`: 平均吞吐量（考虑batch size）

### 4. 显存占用
- `peak_gpu_mem_MB`: 峰值GPU显存（推理时）
- `allocated_MB`: 当前分配显存
- `reserved_MB`: 缓存的显存

### 5. 公平对比机制
- **固定输入集**: 从验证集随机选择N=200张图像（固定种子）
- **列表复用**: 所有模型使用同一 `benchmark_list.txt`
- **避免样本差异**: 确保不同模型在相同输入上测试

## 使用方法

### 基本用法

```bash
# 基准测试单个模型
python scripts/benchmark_model.py \
    --weights results/runs/baseline/seed0/weights/best.pt \
    --imgsz 640 \
    --device 0
```

### 完整参数

```bash
python scripts/benchmark_model.py \
    --weights results/runs/baseline/seed0/weights/best.pt \
    --imgsz 640 \
    --device 0 \
    --warmup 50 \
    --iters 300 \
    --batch 1 \
    --data datasets/openparts/data.yaml \
    --seed 42 \
    --benchmark_size 200
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | - | 模型权重路径（必需） |
| `--imgsz` | 640 | 输入图像尺寸 |
| `--device` | 0 | 设备（0, 1, cpu） |
| `--warmup` | 50 | 预热迭代次数 |
| `--iters` | 300 | 测试迭代次数 |
| `--batch` | 1 | 批次大小 |
| `--data` | datasets/openparts/data.yaml | 数据集配置 |
| `--seed` | 42 | 随机种子（选择图像） |
| `--benchmark_size` | 200 | 基准测试图像数量 |
| `--use_benchmark_list` | False | 使用已有的benchmark_list.txt |

## 公平对比工作流

### 场景：对比6个消融实验模型

```bash
#!/bin/bash
# 批量基准测试脚本

experiments=(
    "baseline"
    "ghost"
    "eca"
    "p2"
    "ghost_eca"
    "ghost_eca_p2"
)

# 第一个模型：创建基准列表
echo "测试模型 1: ${experiments[0]}"
python scripts/benchmark_model.py \
    --weights results/runs/${experiments[0]}/seed0/weights/best.pt \
    --imgsz 640 \
    --device 0 \
    --warmup 50 \
    --iters 300 \
    --batch 1 \
    --benchmark_size 200 \
    --seed 42

# 后续模型：复用基准列表
for exp in "${experiments[@]:1}"; do
    echo ""
    echo "测试模型: $exp"
    python scripts/benchmark_model.py \
        --weights results/runs/$exp/seed0/weights/best.pt \
        --imgsz 640 \
        --device 0 \
        --warmup 50 \
        --iters 300 \
        --batch 1 \
        --use_benchmark_list
done

echo ""
echo "所有模型基准测试完成！"
echo "结果保存在: results/bench/"
```

### 关键点

1. **第一次运行**：不加 `--use_benchmark_list`，自动创建 `benchmark_list.txt`
2. **后续运行**：加上 `--use_benchmark_list`，复用相同的图像列表
3. **固定种子**：使用相同的 `--seed` 确保列表一致（如果重新创建）

## 输出文件

### 输出路径
```
results/bench/
├── benchmark_list.txt         # 固定的测试图像列表（共享）
├── baseline.json              # baseline模型的基准结果
├── ghost.json                 # ghost模型的基准结果
├── eca.json                   # eca模型的基准结果
└── ...
```

### JSON结构

```json
{
  "metadata": {
    "model_name": "baseline",
    "weights": "results/runs/baseline/seed0/weights/best.pt",
    "imgsz": 640,
    "batch": 1,
    "device": "cuda:0",
    "warmup": 50,
    "iters": 300,
    "timestamp": "2026-02-02T...",
    "benchmark_list": "results/bench/benchmark_list.txt",
    "benchmark_size": 200
  },
  "model_complexity": {
    "params": 3157200,
    "params_M": 3.157,
    "gflops": 8.2
  },
  "latency": {
    "mean_ms": 5.23,
    "median_ms": 5.12,
    "p50_ms": 5.12,
    "p95_ms": 6.34,
    "p99_ms": 7.12,
    "std_ms": 0.45,
    "min_ms": 4.89,
    "max_ms": 8.56
  },
  "throughput": {
    "fps_mean": 191.2,
    "fps_p50": 195.3
  },
  "memory": {
    "peak_gpu_mem_MB": 512.5,
    "allocated_MB": 450.2,
    "reserved_MB": 600.0
  }
}
```

## 结果分析

### 对比多个模型

```python
import json
from pathlib import Path
import pandas as pd

# 读取所有结果
results = []
bench_dir = Path("results/bench")

for json_file in bench_dir.glob("*.json"):
    with open(json_file) as f:
        data = json.load(f)
    
    results.append({
        "Model": data["metadata"]["model_name"],
        "Params (M)": data["model_complexity"]["params_M"],
        "GFLOPs": data["model_complexity"]["gflops"],
        "Latency (ms)": data["latency"]["mean_ms"],
        "P95 (ms)": data["latency"]["p95_ms"],
        "FPS": data["throughput"]["fps_mean"],
        "GPU Mem (MB)": data["memory"]["peak_gpu_mem_MB"],
    })

# 创建对比表
df = pd.DataFrame(results)
df = df.sort_values("Params (M)")

print(df.to_markdown(index=False))
```

### 示例输出

```
| Model           | Params (M) | GFLOPs | Latency (ms) | P95 (ms) | FPS    | GPU Mem (MB) |
|-----------------|------------|--------|--------------|----------|--------|--------------|
| ghost_eca_p2    | 2.32       | 6.8    | 4.89         | 5.23     | 204.5  | 480.2        |
| ghost_eca       | 2.14       | 6.5    | 4.67         | 5.01     | 214.1  | 465.3        |
| ghost           | 2.14       | 6.5    | 4.65         | 4.99     | 215.1  | 464.8        |
| baseline        | 3.16       | 8.2    | 5.23         | 5.78     | 191.2  | 512.5        |
| eca             | 3.16       | 8.2    | 5.25         | 5.80     | 190.5  | 513.2        |
| p2              | 3.35       | 9.1    | 5.67         | 6.23     | 176.4  | 542.1        |
```

## 常见问题

### Q1: 为什么要使用固定的基准列表？
**A**: 不同图像的复杂度不同，使用随机图像可能导致：
- 模型A测试到简单图像，延迟低
- 模型B测试到复杂图像，延迟高
- 对比不公平

使用固定列表确保所有模型在相同图像上测试。

### Q2: warmup和iters应该设置多少？
**A**: 推荐值：
- **warmup=50**: GPU需要预热，初始几次推理较慢
- **iters=300**: 足够多的样本获得稳定统计（可以设置更多如1000）

### Q3: 如何测试不同batch size？
**A**: 
```bash
# batch=1（单图推理）
python scripts/benchmark_model.py --weights model.pt --batch 1

# batch=8（批量推理）
python scripts/benchmark_model.py --weights model.pt --batch 8
```

注意：FPS会自动考虑batch size: `fps = 1000 / latency * batch`

### Q4: CPU和GPU结果可以对比吗？
**A**: 不建议直接对比。CPU和GPU架构不同，应该：
- GPU模型间对比
- CPU模型间对比
- 同一模型在CPU vs GPU上对比

### Q5: 如何减少测试时间？
**A**: 调整参数：
```bash
--warmup 10 --iters 100 --benchmark_size 50
```
但统计精度会降低。

## 验收检查

运行以下命令确保功能正常：

```bash
# 1. 单元测试
python scripts/test_benchmark.py

# 2. 测试两个模型（如果有训练好的权重）
python scripts/benchmark_model.py \
    --weights results/runs/baseline/seed0/weights/best.pt \
    --device 0

python scripts/benchmark_model.py \
    --weights results/runs/ghost/seed0/weights/best.pt \
    --device 0 \
    --use_benchmark_list

# 3. 检查输出
ls -lh results/bench/
cat results/bench/baseline.json
cat results/bench/ghost.json

# 4. 验证benchmark_list复用
diff <(jq -r '.metadata.benchmark_list' results/bench/baseline.json) \
     <(jq -r '.metadata.benchmark_list' results/bench/ghost.json)
# 应该相同
```

## 高级用法

### 多GPU测试

```bash
# GPU 0
python scripts/benchmark_model.py --weights model1.pt --device 0

# GPU 1
python scripts/benchmark_model.py --weights model2.pt --device 1
```

### 不同输入尺寸

```bash
# 640x640
python scripts/benchmark_model.py --weights model.pt --imgsz 640

# 1280x1280（高分辨率）
python scripts/benchmark_model.py --weights model.pt --imgsz 1280
```

### 批量统计脚本

保存为 `scripts/analyze_benchmark.py`:

```python
import json
from pathlib import Path
import matplotlib.pyplot as plt

bench_dir = Path("results/bench")
models = {}

for json_file in bench_dir.glob("*.json"):
    with open(json_file) as f:
        data = json.load(f)
    name = data["metadata"]["model_name"]
    models[name] = data

# 绘制Params vs Latency
fig, ax = plt.subplots(figsize=(10, 6))

for name, data in models.items():
    params = data["model_complexity"]["params_M"]
    latency = data["latency"]["mean_ms"]
    ax.scatter(params, latency, s=100, label=name)
    ax.text(params, latency, name, fontsize=9)

ax.set_xlabel("Params (M)")
ax.set_ylabel("Latency (ms)")
ax.set_title("Model Complexity vs Inference Latency")
ax.legend()
ax.grid(True)

plt.savefig("results/bench/comparison.png", dpi=300, bbox_inches="tight")
print("Comparison plot saved to: results/bench/comparison.png")
```

## 相关文档

- [训练运行器](./run_train_one.py)
- [评估运行器](./run_eval_one.py)
- [实验协议](../docs/EXPERIMENT_PROTOCOL.md)
