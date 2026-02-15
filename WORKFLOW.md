# YOLO 消融实验完整工作流程

## 📋 系统概述

本项目提供完整的 YOLO 消融实验自动化工具链，从训练到报告生成一条龙服务。

## 🎯 核心功能

### 1. 单次实验运行
- **训练**: `scripts/run_train_one.py`
- **评估**: `scripts/run_eval_one.py`
- **基准测试**: `scripts/benchmark_model.py`

### 2. 批量实验自动化
- **消融计划**: `experiments/ablation_plan.yaml`
- **自动化运行器**: `scripts/run_ablation.py`

### 3. 结果分析与报告
- **结果汇总**: `scripts/aggregate_results.py`
- **报告生成**: `scripts/make_report.py`

## 🚀 完整工作流程

### 方案 A: 一键运行（推荐）

```bash
# 1. 运行所有消融实验（6实验 × 3种子 = 18任务）
python scripts/run_ablation.py

# 2. 汇总结果
python scripts/aggregate_results.py

# 3. 生成报告
python scripts/make_report.py

# 完成！查看报告
cat results/summary/ablation_report.md
```

### 方案 B: 单步运行（调试用）

```bash
# 1. 训练单个实验
python scripts/run_train_one.py \
    --exp_name baseline \
    --model_yaml ultralytics/cfg/models/v8/yolov8n_baseline.yaml \
    --seed 0 \
    --train_cfg experiments/base_train.yaml

# 2. 评估
python scripts/run_eval_one.py \
    --exp_name baseline \
    --weights results/runs/baseline/seed0/weights/best.pt \
    --seed 0 \
    --split test \
    --eval_cfg experiments/base_eval.yaml

# 3. 基准测试
python scripts/benchmark_model.py \
    --weights results/runs/baseline/seed0/weights/best.pt \
    --imgsz 640 \
    --device 0

# 4. 重复步骤 1-3 完成其他实验和种子...

# 5. 汇总和报告（同方案A）
python scripts/aggregate_results.py
python scripts/make_report.py
```

## 📂 目录结构

```
ultralytics/
├── scripts/                              # 脚本工具
│   ├── run_train_one.py                 # 单次训练
│   ├── run_eval_one.py                  # 单次评估（4种分析）
│   ├── benchmark_model.py               # 性能基准测试
│   ├── run_ablation.py                  # 消融实验自动化
│   ├── aggregate_results.py             # 结果汇总
│   ├── make_report.py                   # 报告生成
│   ├── test_aggregate_report.py         # 测试脚本
│   └── ...
│
├── experiments/                          # 实验配置
│   ├── ablation_plan.yaml               # 消融实验计划
│   ├── base_train.yaml                  # 训练配置模板
│   ├── base_eval.yaml                   # 评估配置模板
│   ├── README_ablation.md               # 消融实验使用指南
│   └── README_aggregate_report.md       # 结果分析使用指南
│
├── results/                              # 实验结果
│   ├── runs/                            # 训练输出
│   │   ├── baseline/
│   │   │   ├── seed0/
│   │   │   │   ├── weights/
│   │   │   │   │   ├── best.pt
│   │   │   │   │   └── last.pt
│   │   │   │   ├── results.csv
│   │   │   │   └── args.yaml
│   │   │   ├── seed1/
│   │   │   └── seed2/
│   │   ├── ghost/
│   │   ├── eca/
│   │   └── ...
│   │
│   ├── evals/                           # 评估输出
│   │   ├── baseline_seed0_test/
│   │   │   ├── metrics.json            # 标准指标
│   │   │   ├── size_metrics.json       # 尺度分析
│   │   │   ├── center_errors.json      # 中心误差
│   │   │   └── failure_cases.csv       # 失败案例
│   │   └── ...
│   │
│   ├── benchmarks/                      # 基准测试
│   │   ├── baseline_seed0_benchmark.json
│   │   └── ...
│   │
│   ├── summary/                         # 汇总报告
│   │   ├── ablation_summary.csv        # CSV汇总表
│   │   ├── ablation_summary.json       # 完整数据
│   │   ├── ablation_report.md          # Markdown报告
│   │   ├── plot_map_vs_fps.png         # 对比图1
│   │   ├── plot_ap_small_vs_fps.png    # 对比图2
│   │   └── plot_center_err_vs_fps.png  # 对比图3
│   │
│   ├── benchmark_list.txt               # 固定测试图像列表
│   └── ablation_failures.log            # 失败记录
│
└── ultralytics/cfg/models/v8/           # 模型配置
    ├── yolov8n_baseline.yaml
    ├── yolov8n_ghost.yaml
    ├── yolov8n_eca.yaml
    ├── yolov8n_p2.yaml
    ├── yolov8n_ghost_eca.yaml
    └── yolov8n_ghost_eca_p2.yaml
```

## 📊 输出指标详解

### 训练输出
- **权重文件**: `best.pt`, `last.pt`
- **训练曲线**: `results.csv` (loss, mAP, etc.)
- **参数快照**: `args.yaml` (完整训练参数)

### 评估输出

#### 1. 标准指标 (`metrics.json`)
```json
{
  "mAP50": 0.752,
  "mAP50-95": 0.543,
  "Precision": 0.812,
  "Recall": 0.731
}
```

#### 2. 尺度分析 (`size_metrics.json`)
```json
{
  "AP_small": 0.321,    // area < 32²
  "AP_medium": 0.654,   // 32² ≤ area < 96²
  "AP_large": 0.812     // area ≥ 96²
}
```

#### 3. 中心误差 (`center_errors.json`)
```json
{
  "mean_error_pixels": 3.45,
  "median_error_pixels": 2.31,
  "max_error_pixels": 15.67
}
```

#### 4. 失败案例 (`failure_cases.csv`)
```csv
image,class,type,confidence,iou,bbox
image001.jpg,person,FP,0.95,0.0,[100,200,150,300]
image002.jpg,car,FN,0.0,0.0,[200,300,250,400]
```

### 基准测试输出 (`benchmark.json`)
```json
{
  "model_params": 3162272,
  "model_gflops": 8.2,
  "latency_mean_ms": 12.34,
  "fps": 81.0,
  "memory_peak_mb": 478.3
}
```

### 汇总报告输出

#### CSV 表格 (`ablation_summary.csv`)
- 每个实验的 mean ± std
- 相对 baseline 的提升百分比
- 可导入 Excel 分析

#### Markdown 报告 (`ablation_report.md`)
- 实验设置（环境、协议、实验列表）
- 主结果表（格式化表格）
- 可视化分析（三张对比图）
- 关键发现（最佳模型）
- 不足分析（自动生成）
- 改进建议（基于数据）

#### 对比图表
- `plot_map_vs_fps.png`: 精度-速度 trade-off
- `plot_ap_small_vs_fps.png`: 小目标检测对比
- `plot_center_err_vs_fps.png`: 定位精度对比

## 🎓 使用场景

### 场景 1: 首次运行完整消融实验

```bash
# 1. 预览执行计划（不实际运行）
python scripts/run_ablation.py --dry-run

# 2. 确认后运行
python scripts/run_ablation.py

# 3. 汇总和报告
python scripts/aggregate_results.py
python scripts/make_report.py
```

### 场景 2: 实验中断后继续

```bash
# 直接重新运行，自动跳过已完成任务
python scripts/run_ablation.py
```

系统会检测：
- 训练权重 (`best.pt`)
- 评估结果 (`metrics.json`)
- 基准测试 (`benchmark.json`)

已完成的任务自动跳过。

### 场景 3: 单个实验调试

```bash
# 只运行 baseline seed0
python scripts/run_ablation.py --exp baseline --seed 0

# 或分步调试
python scripts/run_train_one.py --exp_name baseline --model_yaml ... --seed 0
python scripts/run_eval_one.py --exp_name baseline --weights ... --seed 0 --split test
python scripts/benchmark_model.py --weights ...
```

### 场景 4: 新增实验后更新报告

```bash
# 假设新增了 ghost_eca_p2 实验
python scripts/run_ablation.py --exp ghost_eca_p2

# 重新汇总和报告（自动包含新实验）
python scripts/aggregate_results.py
python scripts/make_report.py
```

### 场景 5: 只重新评估或基准测试

```bash
# 训练已完成，只想重新评估
python scripts/run_ablation.py --skip-train --force

# 只运行基准测试
python scripts/run_ablation.py --skip-train --skip-eval --force
```

## 🔧 配置修改

### 修改训练超参数

编辑 `experiments/base_train.yaml`:
```yaml
epochs: 100       # 训练轮数
batch: 8          # 批大小
lr0: 0.01         # 初始学习率
imgsz: 640        # 图像尺寸
```

### 修改评估设置

编辑 `experiments/base_eval.yaml`:
```yaml
conf: 0.25        # 置信度阈值
iou: 0.7          # NMS IoU 阈值
max_det: 300      # 最大检测数
```

### 添加新实验

编辑 `experiments/ablation_plan.yaml`:
```yaml
experiments:
  - name: my_new_model
    model_yaml: ultralytics/cfg/models/v8/my_model.yaml
    description: My custom modification
    enabled: true
```

### 修改种子

编辑 `experiments/ablation_plan.yaml`:
```yaml
seeds: [0, 1, 2, 3, 4]  # 增加到 5 个种子
```

## 🧪 测试验证

### 单元测试

```bash
# 测试评估功能
python scripts/run_eval_one.py --help  # 查看帮助
# 实际测试需要先训练模型

# 测试基准测试
python scripts/benchmark_model.py --help

# 测试消融自动化
python scripts/test_ablation.py

# 测试结果汇总和报告
python scripts/test_aggregate_report.py
```

### 模拟数据测试

```bash
# 使用模拟数据测试完整流程
python scripts/test_aggregate_report.py
# 会创建模拟数据 → 汇总 → 报告 → 验证输出
```

## 📈 性能优化建议

### 训练加速
- 增大 batch size（如果显存允许）
- 使用混合精度训练（AMP）
- 多 GPU 训练（DDP）

### 评估加速
- 减小 max_det
- 提高 conf 阈值
- 使用 FP16 推理

### 批量实验优化
- 使用 `--skip-*` 跳过不需要的阶段
- 利用断点续跑避免重复计算
- 禁用不需要的实验（`enabled: false`）

## 🐛 常见问题

### Q1: 内存不足
**症状**: CUDA out of memory

**解决方案**:
```bash
# 减小 batch size
python scripts/run_train_one.py --batch 4  # 默认 8

# 或修改 base_train.yaml
batch: 4
```

### Q2: 某个实验失败
**症状**: 部分实验成功，部分失败

**解决方案**:
```bash
# 1. 查看失败日志
cat results/ablation_failures.log

# 2. 单独运行失败的实验
python scripts/run_ablation.py --exp failed_exp --seed 0

# 3. 修复问题后，重新运行（自动跳过成功的）
python scripts/run_ablation.py
```

### Q3: 图表不显示中文
**症状**: 图表标签是方框

**解决方案**:
```python
# 安装中文字体
# Windows: 已默认支持 SimHei
# Linux: sudo apt-get install fonts-noto-cjk

# 或修改 make_report.py 使用其他字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
```

### Q4: 汇总时提示未找到结果
**症状**: `❌ 未找到任何实验结果`

**解决方案**:
```bash
# 确认实验已运行
ls results/runs/
ls results/evals/
ls results/benchmarks/

# 如果为空，先运行实验
python scripts/run_ablation.py
```

## 📚 文档索引

- **消融实验自动化**: `experiments/README_ablation.md`
- **结果分析与报告**: `experiments/README_aggregate_report.md`
- **训练脚本**: `scripts/run_train_one.py --help`
- **评估脚本**: `scripts/run_eval_one.py --help`
- **基准测试**: `scripts/benchmark_model.py --help`

## 🎉 快速检查清单

完成实验后，检查以下文件：

- [ ] 训练权重: `results/runs/*/seed*/weights/best.pt`
- [ ] 评估指标: `results/evals/*_test/metrics.json`
- [ ] 基准测试: `results/benchmarks/*_benchmark.json`
- [ ] CSV 汇总: `results/summary/ablation_summary.csv`
- [ ] Markdown 报告: `results/summary/ablation_report.md`
- [ ] 对比图表: `results/summary/plot_*.png` (3张)

全部 ✅ → 实验完成，可撰写论文！

---

**Happy experimenting!** 🚀
