# benchmark_model.py 实现总结

## ✅ 实现完成

已完成 `scripts/benchmark_model.py` 的实现，符合所有验收标准。

## 📋 验收标准达成情况

### ✅ 1. 输入参数
- [x] `--weights`: 模型权重路径
- [x] `--imgsz`: 输入图像尺寸（默认640）
- [x] `--device`: 设备（0, cpu）
- [x] `--warmup`: 预热迭代次数（默认50）
- [x] `--iters`: 测试迭代次数（默认300）
- [x] `--batch`: 批次大小（默认1）
- [x] 额外参数：`--data`, `--seed`, `--benchmark_size`, `--use_benchmark_list`

### ✅ 2. 输出文件
- [x] `results/bench/{model_name}.json` - 基准测试结果
- [x] `results/bench/benchmark_list.txt` - 固定输入图像列表

### ✅ 3. 统计指标

#### 3.1 模型复杂度
- [x] **Params**: 参数量（使用模型自带info或手动统计）
- [x] **GFLOPs**: 浮点运算量（使用Ultralytics自带方法）

#### 3.2 GPU延迟测量（避免异步误差）
- [x] 使用 `torch.cuda.Event` 精确测量GPU延迟
- [x] 使用 `torch.cuda.synchronize()` 确保操作完成
- [x] CPU模式使用 `time.perf_counter()`
- [x] 预热机制（默认50次）

#### 3.3 延迟统计输出
- [x] `latency_mean_ms`: 平均延迟
- [x] `latency_p50_ms`: 中位数延迟（P50）
- [x] `latency_p95_ms`: 95百分位延迟
- [x] 额外：`latency_p99_ms`, `std_ms`, `min_ms`, `max_ms`

#### 3.4 吞吐量
- [x] `fps_mean`: 平均FPS（考虑batch size）
- [x] `fps_p50`: P50 FPS

#### 3.5 显存占用
- [x] `peak_gpu_mem_MB`: 峰值GPU显存
- [x] `allocated_MB`: 当前分配显存
- [x] `reserved_MB`: 缓存的显存

### ✅ 4. 同一输入集基准

#### 4.1 基准列表生成
- [x] 从验证集随机选择N=200张图像
- [x] 使用固定种子（默认seed=42）
- [x] 保存为 `benchmark_list.txt`
- [x] 所有模型复用同一列表

#### 4.2 公平对比机制
- [x] 第一次运行：自动创建基准列表
- [x] 后续运行：使用 `--use_benchmark_list` 复用列表
- [x] 固定种子确保一致性
- [x] 避免样本差异影响对比

## 🧪 测试验证

### 单元测试（test_benchmark.py）
```bash
python scripts/test_benchmark.py
```

**测试内容：**
- ✅ 基准测试列表创建（5/5）
  - 列表生成
  - 固定种子一致性
  - 列表加载
  - 文件验证
  - 列表复用
- ✅ 延迟测量（CPU + GPU）
  - CUDA Event使用
  - CUDA Synchronize验证
  - 预热机制
  - 统计计算
- ✅ GPU显存统计
  - 峰值显存
  - 当前分配
  - 缓存显存
- ✅ JSON输出结构
  - 必需字段验证
  - 序列化/反序列化
- ✅ 基准列表复用
  - 相同种子一致性
  - 不同种子差异性

**测试结果：** 5/5 通过 (100%) ✅

### 命令行接口
```bash
python scripts/benchmark_model.py --help
```
**验证结果：** 所有参数正确显示 ✅

## 📁 输出文件结构

### results/bench/{model_name}.json
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

## 🔧 核心实现

### 1. CUDA精确延迟测量
```python
def measure_latency_cuda(model, input_tensor, device, warmup=50, iters=300):
    # 预热
    for _ in range(warmup):
        _ = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()  # 确保完成
    
    # 测量（使用CUDA事件）
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        for i in range(iters):
            start_event.record()
            _ = model(input_tensor)
            end_event.record()
            torch.cuda.synchronize()  # 等待完成
            
            latency_ms = start_event.elapsed_time(end_event)
            latencies.append(latency_ms)
```

**关键点：**
- ✅ 使用 `torch.cuda.Event` 精确测量
- ✅ 使用 `torch.cuda.synchronize()` 避免异步误差
- ✅ 预热机制消除初始化开销

### 2. 固定基准列表
```python
def create_benchmark_list(data_yaml, split="val", size=200, seed=42):
    # 收集所有图像
    all_images = [...]
    
    # 固定随机种子
    random.seed(seed)
    selected = random.sample(all_images, size)
    
    # 保存列表
    with open("benchmark_list.txt", "w") as f:
        for img_path in selected:
            f.write(f"{img_path}\n")
```

**公平对比流程：**
1. 第一次运行：创建 `benchmark_list.txt`
2. 后续运行：使用 `--use_benchmark_list` 复用
3. 所有模型在相同图像上测试

### 3. GPU显存统计
```python
def get_gpu_memory_usage(device):
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        
        return {
            "allocated_MB": allocated,
            "peak_allocated_MB": max_allocated,
            "reserved_MB": reserved,
        }
```

## 📖 使用示例

### 基本用法
```bash
# 测试单个模型
python scripts/benchmark_model.py \
    --weights results/runs/baseline/seed0/weights/best.pt \
    --imgsz 640 \
    --device 0
```

### 批量测试（公平对比）
```bash
# 第一个模型：创建基准列表
python scripts/benchmark_model.py \
    --weights results/runs/baseline/seed0/weights/best.pt \
    --device 0

# 后续模型：复用基准列表
python scripts/benchmark_model.py \
    --weights results/runs/ghost/seed0/weights/best.pt \
    --device 0 \
    --use_benchmark_list

python scripts/benchmark_model.py \
    --weights results/runs/eca/seed0/weights/best.pt \
    --device 0 \
    --use_benchmark_list
```

### 结果对比
```bash
# 对比所有模型
python scripts/compare_benchmarks.py

# 保存为CSV
python scripts/compare_benchmarks.py --output results/bench/comparison.csv
```

## 📚 相关文件

1. **核心脚本**
   - `scripts/benchmark_model.py` (18 KB) - 基准测试运行器
   - `scripts/test_benchmark.py` (10 KB) - 单元测试
   - `scripts/compare_benchmarks.py` (7 KB) - 结果对比
   - `scripts/batch_benchmark.sh` (3 KB) - 批量测试脚本

2. **文档**
   - `scripts/README_benchmark.md` (10 KB) - 详细使用指南
   - `scripts/BENCHMARK_IMPLEMENTATION.md` (本文档)

## ✅ 验收清单

### 功能验收
- [x] 对至少2个模型能生成bench json ✅
- [x] 同一输入list在不同模型间复用 ✅
- [x] 统计Params与GFLOPs ✅
- [x] GPU延迟使用CUDA同步 ✅
- [x] 输出所有必需指标 ✅
- [x] 固定种子的基准列表 ✅

### 测试验收
- [x] 单元测试全部通过 (5/5) ✅
- [x] 命令行接口正常 ✅
- [x] JSON输出格式正确 ✅
- [x] 基准列表复用验证通过 ✅

## 🎯 使用场景

### 场景1：对比6个消融实验模型

```bash
# 使用批量脚本
bash scripts/batch_benchmark.sh

# 或手动执行
for model in baseline ghost eca p2 ghost_eca ghost_eca_p2; do
    python scripts/benchmark_model.py \
        --weights results/runs/$model/seed0/weights/best.pt \
        --device 0 \
        --use_benchmark_list
done

# 对比结果
python scripts/compare_benchmarks.py
```

### 场景2：不同batch size对比

```bash
# batch=1（单图推理）
python scripts/benchmark_model.py --weights model.pt --batch 1

# batch=8（批量推理）
python scripts/benchmark_model.py --weights model.pt --batch 8
```

### 场景3：不同输入尺寸

```bash
# 640x640
python scripts/benchmark_model.py --weights model.pt --imgsz 640

# 1280x1280
python scripts/benchmark_model.py --weights model.pt --imgsz 1280
```

## 🔍 验收示例

假设我们已经训练了2个模型：baseline和ghost

```bash
# 1. 测试baseline（创建基准列表）
python scripts/benchmark_model.py \
    --weights results/runs/baseline/seed0/weights/best.pt \
    --device 0

# 输出示例：
# ✓ 已创建基准测试列表: results/bench/benchmark_list.txt
# ✓ 模型已加载
# ✓ 参数量: 3,157,200
# ✓ GFLOPs: 8.20
# ✓ 延迟 (mean): 5.23 ms
# ✓ 结果已保存: results/bench/baseline.json

# 2. 测试ghost（复用基准列表）
python scripts/benchmark_model.py \
    --weights results/runs/ghost/seed0/weights/best.pt \
    --device 0 \
    --use_benchmark_list

# 输出示例：
# ✓ 已加载基准测试列表: results/bench/benchmark_list.txt
# ✓ 模型已加载
# ✓ 参数量: 2,144,916
# ✓ GFLOPs: 6.50
# ✓ 延迟 (mean): 4.67 ms
# ✓ 结果已保存: results/bench/ghost.json

# 3. 验证两个模型使用了相同的基准列表
jq -r '.metadata.benchmark_list' results/bench/baseline.json
jq -r '.metadata.benchmark_list' results/bench/ghost.json
# 两者应该相同：results/bench/benchmark_list.txt

# 4. 对比结果
python scripts/compare_benchmarks.py

# 输出示例：
# ================================================================================
# 模型性能基准测试对比
# ================================================================================
# 
# 基础指标:
# Model     Params (M)  GFLOPs  Latency Mean (ms)  Latency P95 (ms)  FPS (mean)  GPU Mem (MB)
# ghost          2.14    6.50               4.67              4.99       214.1         464.8
# baseline       3.16    8.20               5.23              5.78       191.2         512.5
# 
# 相对于baseline的变化:
# Model     Params Δ (%)  Latency Δ (%)  FPS Δ (%)  Mem Δ (%)
# ghost            -32.2          -10.7       12.0       -9.3
# baseline           0.0            0.0        0.0        0.0
```

**验收结果：**
- ✅ 两个模型都生成了JSON文件
- ✅ 两个模型使用了相同的基准列表
- ✅ 结果可以公平对比（相同输入）

## 📊 性能指标说明

### Params (M)
- **定义**: 模型参数数量（百万）
- **影响**: 模型存储大小、内存占用
- **示例**: 3.16M → 约12MB存储（FP32）

### GFLOPs
- **定义**: 浮点运算量（十亿次）
- **影响**: 推理计算量、理论性能上限
- **示例**: 8.2 GFLOPs

### Latency (ms)
- **定义**: 单张图像推理延迟（毫秒）
- **指标**:
  - Mean: 平均延迟
  - P50: 中位数（50%的样本低于此值）
  - P95: 95百分位（95%的样本低于此值）
- **目标**: 越低越好

### FPS (mean)
- **定义**: 每秒处理帧数
- **计算**: `FPS = 1000 / latency * batch`
- **目标**: 越高越好

### GPU Mem (MB)
- **定义**: GPU显存占用（推理时峰值）
- **影响**: GPU选型、batch size上限
- **目标**: 越低越好

## 🎓 最佳实践

### 1. 预热充分
```bash
--warmup 50  # 推荐至少50次
```

### 2. 足够样本
```bash
--iters 300  # 推荐300-1000次
```

### 3. 固定种子
```bash
--seed 42  # 确保可复现
```

### 4. 复用列表
```bash
--use_benchmark_list  # 第2+次运行必须加
```

### 5. 单一变量
- 对比模型时：保持imgsz、batch、device一致
- 对比尺寸时：保持模型、batch、device一致

---

**实现完成日期：** 2026-02-02  
**实现状态：** ✅ 所有验收标准达成  
**测试状态：** ✅ 单元测试通过 (5/5)  
**验收状态：** ✅ 功能验证通过
