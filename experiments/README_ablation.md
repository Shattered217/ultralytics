# 消融实验自动化使用指南

## 📋 概述

本指南介绍如何使用消融实验自动化系统完成 YOLO 模型的全流程消融实验，包括：
- **训练**：多配置 × 多种子的批量训练
- **评估**：val 和 test 数据集评估
- **基准测试**：性能指标测量
- **断点续跑**：自动检测已完成任务并跳过
- **失败记录**：记录失败情况便于调试

## 🎯 实验计划

实验计划定义在 `experiments/ablation_plan.yaml`：

```yaml
experiments:
  - name: baseline          # 基线模型（标准 YOLOv8n）
  - name: ghost             # Ghost 卷积（参数量 -32%）
  - name: eca               # ECA 注意力机制
  - name: p2                # P2 层（小目标检测）
  - name: ghost_eca         # Ghost + ECA 组合
  - name: ghost_eca_p2      # 完整改进模型

seeds: [0, 1, 2]            # 3 个随机种子

总任务数: 6 实验 × 3 种子 = 18 个任务
```

每个任务执行流程：
```
训练 (run_train_one.py)
  ↓
评估-val (run_eval_one.py)
  ↓
评估-test (run_eval_one.py)
  ↓
基准测试 (benchmark_model.py)
```

## 🚀 快速开始

### 1. 预览执行计划

在实际运行前，先预览将要执行的命令：

```bash
python scripts/run_ablation.py --dry-run
```

这会显示：
- 将执行的所有实验和种子组合
- 每个任务的具体命令
- 断点续跑检查结果
- 不会实际运行，仅预览

### 2. 运行所有实验

确认执行计划后，运行所有 18 个任务：

```bash
python scripts/run_ablation.py
```

系统会：
- ✅ 按顺序执行每个 (实验, 种子) 组合
- ✅ 每个任务完成后自动进行评估和基准测试
- ✅ 自动跳过已完成的任务（断点续跑）
- ✅ 记录失败情况到 `results/ablation_failures.log`
- ✅ 显示实时进度和总结

### 3. 运行特定实验

**只运行 baseline 模型，种子 0：**
```bash
python scripts/run_ablation.py --exp baseline --seed 0
```

**只运行 ghost 模型，所有种子：**
```bash
python scripts/run_ablation.py --exp ghost
```

**运行所有实验，但只用种子 1：**
```bash
python scripts/run_ablation.py --seed 1
```

## 🔧 高级选项

### 断点续跑

系统会自动检测已完成的任务并跳过。如果实验中断，直接重新运行相同命令即可继续：

```bash
# 第一次运行（完成了 3 个任务后中断）
python scripts/run_ablation.py

# 重新运行（自动从第 4 个任务开始）
python scripts/run_ablation.py
```

检测逻辑：
- **训练**：检查 `results/runs/{exp_name}/seed{seed}/weights/best.pt`
- **评估-val**：检查 `results/evals/{exp_name}_seed{seed}_val/metrics.json`
- **评估-test**：检查 `results/evals/{exp_name}_seed{seed}_test/metrics.json`
- **基准测试**：检查 `results/benchmarks/{exp_name}_seed{seed}_benchmark.json`

### 强制重新运行

忽略已有输出，强制重新运行：

```bash
python scripts/run_ablation.py --force
```

### 跳过特定阶段

**只运行训练，跳过评估和基准测试：**
```bash
python scripts/run_ablation.py --skip-eval --skip-benchmark
```

**只运行评估，跳过训练和基准测试：**
```bash
python scripts/run_ablation.py --skip-train --skip-benchmark
```

**只运行基准测试：**
```bash
python scripts/run_ablation.py --skip-train --skip-eval
```

## 📊 输出结构

完成后，结果文件组织如下：

```
results/
├── runs/                                # 训练输出
│   ├── baseline/
│   │   ├── seed0/
│   │   │   ├── weights/
│   │   │   │   ├── best.pt           # 最佳权重
│   │   │   │   └── last.pt           # 最后权重
│   │   │   ├── results.csv           # 训练指标
│   │   │   └── args.yaml             # 训练参数快照
│   │   ├── seed1/
│   │   └── seed2/
│   ├── ghost/
│   ├── eca/
│   ├── p2/
│   ├── ghost_eca/
│   └── ghost_eca_p2/
│
├── evals/                              # 评估输出
│   ├── baseline_seed0_val/
│   │   ├── metrics.json              # 标准指标 (mAP50, mAP50-95, P, R)
│   │   ├── size_metrics.json         # 尺度分析 (small/medium/large)
│   │   ├── center_errors.json        # 中心点误差统计
│   │   ├── failure_cases_fp.json     # Top-K 假阳性
│   │   ├── failure_cases_fn.json     # Top-K 假阴性
│   │   └── failure_cases.csv         # 失败案例明细
│   ├── baseline_seed0_test/
│   ├── baseline_seed1_val/
│   └── ...
│
├── benchmarks/                         # 基准测试输出
│   ├── baseline_seed0_benchmark.json # 性能指标 (参数量, GFLOPs, 延迟, 内存)
│   ├── baseline_seed1_benchmark.json
│   └── ...
│
├── ablation_failures.log               # 失败记录 (如有)
└── benchmark_list.txt                  # 固定测试图像列表（公平对比）
```

## 📈 结果分析

### 1. 查看标准指标

```bash
# 查看 baseline seed0 在 val 集上的指标
cat results/evals/baseline_seed0_val/metrics.json
```

输出示例：
```json
{
  "mAP50": 0.752,
  "mAP50-95": 0.543,
  "Precision": 0.812,
  "Recall": 0.731,
  "num_predictions": 1234,
  "num_ground_truths": 1150
}
```

### 2. 查看尺度分析

```bash
# 查看不同尺度目标的 AP
cat results/evals/baseline_seed0_val/size_metrics.json
```

输出示例：
```json
{
  "AP_small": 0.321,    # 小目标 (area < 32²)
  "AP_medium": 0.654,   # 中等目标 (32² ≤ area < 96²)
  "AP_large": 0.812,    # 大目标 (area ≥ 96²)
  "num_small": 456,
  "num_medium": 512,
  "num_large": 182
}
```

### 3. 查看中心点误差

```bash
# 查看中心点定位误差统计
cat results/evals/baseline_seed0_val/center_errors.json
```

输出示例：
```json
{
  "mean_error_pixels": 3.45,
  "median_error_pixels": 2.31,
  "max_error_pixels": 15.67,
  "mean_error_relative": 0.0234,  # 相对于 bbox 宽度
  "num_matched": 1050
}
```

### 4. 查看基准测试结果

```bash
# 查看模型性能指标
cat results/benchmarks/baseline_seed0_benchmark.json
```

输出示例：
```json
{
  "model_params": 3162272,
  "model_gflops": 8.2,
  "latency_mean_ms": 12.34,
  "latency_std_ms": 0.56,
  "fps": 81.0,
  "memory_allocated_mb": 245.6,
  "memory_reserved_mb": 512.0,
  "memory_peak_mb": 478.3
}
```

### 5. 分析失败情况

如果有任务失败，检查失败日志：

```bash
cat results/ablation_failures.log
```

输出示例：
```
2024-01-15 14:32:10 | ghost | seed1 | train | CUDA out of memory
2024-01-15 15:45:23 | eca | seed2 | eval_test | File not found: best.pt
```

## 🔍 常见使用场景

### 场景 1：首次运行完整实验

```bash
# 1. 预览
python scripts/run_ablation.py --dry-run

# 2. 确认后运行
python scripts/run_ablation.py
```

### 场景 2：实验中断后续跑

```bash
# 直接重新运行，自动跳过已完成任务
python scripts/run_ablation.py
```

### 场景 3：单个实验调试

```bash
# 只运行 baseline seed0，方便快速调试
python scripts/run_ablation.py --exp baseline --seed 0
```

### 场景 4：只重新评估

```bash
# 训练已完成，只想重新评估
python scripts/run_ablation.py --skip-train --force
```

### 场景 5：修改实验计划

1. 编辑 `experiments/ablation_plan.yaml`
2. 禁用不需要的实验（设置 `enabled: false`）
3. 运行：

```bash
python scripts/run_ablation.py
```

## ⚙️ 配置修改

### 修改训练超参数

编辑 `experiments/base_train.yaml`：

```yaml
epochs: 100          # 训练轮数
batch: 8             # 批大小
imgsz: 640           # 图像尺寸
lr0: 0.01            # 初始学习率
weight_decay: 0.0005 # 权重衰减
```

### 修改评估设置

编辑 `experiments/base_eval.yaml`：

```yaml
conf: 0.25           # 置信度阈值
iou: 0.7             # NMS IoU 阈值
max_det: 300         # 每张图最大检测数
```

### 修改基准测试参数

编辑 `experiments/ablation_plan.yaml` 中的 `benchmark_config`：

```yaml
benchmark_config:
  imgsz: 640         # 测试图像尺寸
  warmup: 50         # 预热迭代数
  iters: 300         # 测试迭代数
  batch: 1           # 批大小（推理时通常为 1）
  use_benchmark_list: true  # 使用固定图像列表（公平对比）
```

### 添加新实验

编辑 `experiments/ablation_plan.yaml`：

```yaml
experiments:
  # ... 已有实验 ...
  
  - name: my_new_experiment
    model_yaml: ultralytics/cfg/models/v8/yolov8n_my_model.yaml
    description: My new model variant
    enabled: true
```

## 🐛 故障排查

### 问题 1：CUDA Out of Memory

**症状**：训练失败，报错 "CUDA out of memory"

**解决方案**：
1. 减小批大小：编辑 `experiments/base_train.yaml`，将 `batch: 8` 改为 `batch: 4`
2. 减小图像尺寸：将 `imgsz: 640` 改为 `imgsz: 512`
3. 或运行时覆盖：
   ```bash
   python scripts/run_train_one.py --exp_name baseline --model_yaml ... --batch 4
   ```

### 问题 2：权重文件不存在

**症状**：评估失败，报错 "File not found: best.pt"

**解决方案**：
1. 检查训练是否成功完成
2. 查看训练日志：`results/runs/{exp_name}/seed{seed}/`
3. 如果训练失败，先修复训练问题

### 问题 3：断点续跑不生效

**症状**：重新运行时没有跳过已完成任务

**解决方案**：
1. 检查 `experiments/ablation_plan.yaml` 中 `resume: true`
2. 确认输出文件存在：
   ```bash
   ls results/runs/baseline/seed0/weights/best.pt
   ls results/evals/baseline_seed0_val/metrics.json
   ```
3. 如需强制重新运行，使用 `--force` 选项

### 问题 4：部分实验失败

**症状**：部分任务成功，部分失败

**解决方案**：
1. 查看失败日志：`cat results/ablation_failures.log`
2. 根据错误信息修复问题
3. 重新运行（自动跳过成功的任务）：
   ```bash
   python scripts/run_ablation.py
   ```

## 📝 命令行参数完整列表

```bash
python scripts/run_ablation.py [OPTIONS]

选项：
  --plan PLAN              实验计划 YAML 文件
                           默认: experiments/ablation_plan.yaml
                           
  --dry-run                预览执行计划但不实际运行
                           用于检查命令是否正确
                           
  --exp EXP                只运行指定实验
                           例如: --exp baseline
                           
  --seed SEED              只运行指定种子
                           例如: --seed 0
                           
  --skip-train             跳过训练阶段
                           用于已有权重，只想评估
                           
  --skip-eval              跳过评估阶段
                           用于只想训练或基准测试
                           
  --skip-benchmark         跳过基准测试阶段
                           用于快速完成训练和评估
                           
  --force                  强制重新运行
                           忽略已有输出，全部重新执行
                           
  -h, --help               显示帮助信息
```

## 📞 技术支持

**测试验证：**
```bash
# 运行自动化测试套件
python scripts/test_ablation.py
```

预期输出：
```
✅ 所有测试通过！
  ✓ 实验计划加载和解析
  ✓ 断点续跑检测
  ✓ 命令生成 (训练/评估/基准测试)
  ✓ Dry-run 模式
```

**相关文件：**
- 实验计划：`experiments/ablation_plan.yaml`
- 训练配置：`experiments/base_train.yaml`
- 评估配置：`experiments/base_eval.yaml`
- 自动化脚本：`scripts/run_ablation.py`
- 测试脚本：`scripts/test_ablation.py`

---

## 🎓 最佳实践

1. **运行前先预览**：使用 `--dry-run` 检查执行计划
2. **小范围测试**：先用 `--exp baseline --seed 0` 测试单个任务
3. **定期检查**：监控 `results/ablation_failures.log` 发现问题
4. **合理配置**：根据 GPU 内存调整 batch size
5. **利用断点续跑**：实验中断不要慌，重新运行即可继续

Happy experimenting! 🚀
