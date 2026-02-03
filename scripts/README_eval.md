# 评估运行器使用说明

## 概述

`scripts/run_eval_one.py` 是严格可追踪的评估运行器，用于在验证集或测试集上评估训练好的模型。

## 功能特性

### 1. 标准指标（Ultralytics）
- mAP50：IoU=0.5 的平均精度
- mAP50-95：IoU=0.5:0.95 的平均精度
- Precision（精确率）
- Recall（召回率）
- Box Loss、Cls Loss、DFL Loss

### 2. 按目标尺寸分桶的AP统计
根据 bbox 面积（像素²）将 GT 分为三组：
- **Small**: < 32² = 1024 px²
- **Medium**: 32² ~ 96² = 1024 ~ 9216 px²
- **Large**: > 96² = 9216 px²

为每组单独计算 AP50。

### 3. 中心点定位误差
- 使用 **IoU >= 0.5** 做一对一匹配（贪心算法）
- 对匹配上的框计算预测中心点与 GT 中心点的欧氏距离（像素）
- 输出统计：mean、median、p95

**匹配算法：贪心匹配**
1. 计算所有预测框与 GT 框的 IoU 矩阵
2. 从高 IoU 到低 IoU 逐个匹配
3. 已匹配的框不再参与后续匹配
4. 过滤掉 IoU < 0.5 的匹配

### 4. 失败案例导出
- **可视化**：保存 Top-K FP 和 FN 的图像到 `results/runs/{exp_name}/seed{seed}/cases/`
  - 绿色框：True Positive（匹配的预测）
  - 红色框：False Positive（未匹配的预测）
  - 蓝色框：False Negative（未匹配的 GT）
- **CSV摘要**：`cases_summary.csv` 包含图片名、类别、置信度、IoU、错误类型

## 输出文件

```
results/runs/{exp_name}/seed{seed}/
├── eval_val.json              # 验证集评估结果（完整JSON）
├── eval_test.json             # 测试集评估结果（如有）
├── cases/                     # 失败案例可视化
│   ├── fp_000_image1.jpg
│   ├── fp_001_image2.jpg
│   └── ...
└── cases_summary.csv          # 失败案例摘要表
```

### JSON 输出结构

```json
{
  "metadata": {
    "exp_name": "baseline",
    "weights": "results/runs/baseline/seed0/weights/best.pt",
    "seed": 0,
    "split": "val",
    "timestamp": "2026-02-02T..."
  },
  "standard_metrics": {
    "mAP50": 0.7123,
    "mAP50-95": 0.5234,
    "precision": 0.8012,
    "recall": 0.6789
  },
  "size_based_ap": {
    "small": {"ap50": 0.6234, "count": 100},
    "medium": {"ap50": 0.7456, "count": 200},
    "large": {"ap50": 0.8123, "count": 150},
    "thresholds": [32, 96]
  },
  "center_localization_error": {
    "mean": 5.23,
    "median": 4.12,
    "p95": 12.34,
    "count": 450,
    "algorithm": "greedy_matching"
  },
  "failure_cases": {
    "top_k": 20,
    "cases_dir": "results/runs/baseline/seed0/cases",
    "summary_csv": "results/runs/baseline/seed0/cases_summary.csv",
    "total_fp": 15,
    "total_fn": 8
  }
}
```

## 使用方法

### 基本用法

```bash
# 在验证集上评估
python scripts/run_eval_one.py \
    --exp_name baseline \
    --weights results/runs/baseline/seed0/weights/best.pt \
    --seed 0 \
    --split val

# 在测试集上评估
python scripts/run_eval_one.py \
    --exp_name baseline \
    --weights results/runs/baseline/seed0/weights/best.pt \
    --seed 0 \
    --split test
```

### 完整参数

```bash
python scripts/run_eval_one.py \
    --exp_name baseline \
    --weights results/runs/baseline/seed0/weights/best.pt \
    --seed 0 \
    --split val \
    --eval_cfg experiments/base_eval.yaml \
    --data datasets/openparts/data.yaml \
    --top_k 20 \
    --size_thresholds 32 96
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--exp_name` | ✅ | - | 实验名称（与训练时一致） |
| `--weights` | ✅ | - | 模型权重路径（best.pt 或 last.pt） |
| `--seed` | ✅ | - | 随机种子（用于输出路径） |
| `--split` | ❌ | `val` | 数据集split（`val` 或 `test`） |
| `--eval_cfg` | ❌ | `experiments/base_eval.yaml` | 评估配置文件 |
| `--data` | ❌ | - | 数据集配置（覆盖 eval_cfg 中的 data） |
| `--top_k` | ❌ | 20 | 导出的失败案例数量 |
| `--size_thresholds` | ❌ | `32 96` | 尺寸分桶阈值（像素） |

## 评估配置文件

`experiments/base_eval.yaml` 包含所有评估参数：

```yaml
data: datasets/openparts/data.yaml
batch: 16
imgsz: 640
conf: 0.001      # 低阈值以获取完整 mAP 曲线
iou: 0.6         # NMS IoU 阈值
device: 0
workers: 8
```

## 完整评估工作流

### 1. 训练模型

```bash
python scripts/run_train_one.py \
    --exp_name baseline \
    --model_yaml ultralytics/cfg/models/v8/yolov8n_baseline.yaml \
    --seed 0
```

### 2. 在验证集上评估

```bash
python scripts/run_eval_one.py \
    --exp_name baseline \
    --weights results/runs/baseline/seed0/weights/best.pt \
    --seed 0 \
    --split val
```

### 3. 在测试集上评估（如有）

```bash
python scripts/run_eval_one.py \
    --exp_name baseline \
    --weights results/runs/baseline/seed0/weights/best.pt \
    --seed 0 \
    --split test
```

### 4. 查看结果

```bash
# 查看 JSON 结果
cat results/runs/baseline/seed0/eval_val.json

# 查看失败案例
ls results/runs/baseline/seed0/cases/

# 查看案例摘要
cat results/runs/baseline/seed0/cases_summary.csv
```

## 批量评估示例

评估所有消融实验（E0-E7，seeds 0-2）：

```bash
#!/bin/bash
experiments=(
    "baseline:ultralytics/cfg/models/v8/yolov8n_baseline.yaml"
    "ghost:ultralytics/cfg/models/v8/yolov8n_ghost.yaml"
    "eca:ultralytics/cfg/models/v8/yolov8n_eca.yaml"
    "p2:ultralytics/cfg/models/v8/yolov8n_p2.yaml"
    "ghost_eca:ultralytics/cfg/models/v8/yolov8n_ghost_eca.yaml"
    "ghost_eca_p2:ultralytics/cfg/models/v8/yolov8n_ghost_eca_p2.yaml"
)

for exp in "${experiments[@]}"; do
    IFS=':' read -r name model <<< "$exp"
    for seed in 0 1 2; do
        weights="results/runs/${name}/seed${seed}/weights/best.pt"
        
        # 验证集评估
        python scripts/run_eval_one.py \
            --exp_name $name \
            --weights $weights \
            --seed $seed \
            --split val
        
        # 测试集评估（如果数据集有test split）
        python scripts/run_eval_one.py \
            --exp_name $name \
            --weights $weights \
            --seed $seed \
            --split test
    done
done
```

## 验证清单

运行以下检查确保评估正确：

```bash
# 1. 单元测试
python scripts/test_eval_runner.py

# 2. 查看帮助
python scripts/run_eval_one.py --help

# 3. 检查输出文件
exp_name="baseline"
seed=0
echo "检查输出文件..."
ls -lh results/runs/${exp_name}/seed${seed}/eval_val.json
ls -lh results/runs/${exp_name}/seed${seed}/cases/
ls -lh results/runs/${exp_name}/seed${seed}/cases_summary.csv

# 4. 验证JSON结构
python -c "
import json
with open('results/runs/${exp_name}/seed${seed}/eval_val.json') as f:
    data = json.load(f)
    print('✓ standard_metrics:', list(data['standard_metrics'].keys()))
    print('✓ size_based_ap:', list(data['size_based_ap'].keys()))
    print('✓ center_localization_error:', list(data['center_localization_error'].keys()))
    print('✓ failure_cases:', list(data['failure_cases'].keys()))
"
```

## 常见问题

### Q1: 中心点误差统计为空？
**A:** 检查是否有匹配的框（IoU >= 0.5）。如果模型性能很差，可能没有足够的匹配。

### Q2: 尺寸分桶AP如何自定义？
**A:** 使用 `--size_thresholds T1 T2` 参数，例如 `--size_thresholds 40 80`。

### Q3: 失败案例数量不足 Top-K？
**A:** 如果 FP/FN 数量少于 `--top_k`，只会导出实际存在的案例。

### Q4: 评估速度慢？
**A:** 增加 `batch` 大小（修改 `experiments/base_eval.yaml`）或减少 `--top_k`。

## 相关文档

- [训练运行器](./run_train_one.py)
- [实验协议](../docs/EXPERIMENT_PROTOCOL.md)
- [数据集统计](./compute_dataset_stats.py)
- [配置加载](./load_config.py)
