# 实验复现协议 (Experiment Reproducibility Protocol)

本文档定义了 Ultralytics 改进实验的标准化流程，确保所有科研实验的可重复性与可比性。

---

## 1. 固定变量列表 (Fixed Variables)

为了保证实验的公平性和可对比性，以下变量在所有实验中保持固定：

### 1.1 数据划分 (Data Split)
- **训练集/验证集/测试集比例**: 70% / 20% / 10%
- **数据集版本**: 记录数据集的具体版本或日期
- **数据增强策略**: 
  - Mosaic: 1.0
  - MixUp: 0.0
  - Degrees: 0.0
  - Translate: 0.1
  - Scale: 0.5
  - Shear: 0.0
  - Perspective: 0.0
  - HSV-H: 0.015
  - HSV-S: 0.7
  - HSV-V: 0.4
  - Flipud: 0.0
  - Fliplr: 0.5
- **图像尺寸**: 640×640 (标准输入)

### 1.2 训练超参数 (Training Hyperparameters)
- **Epochs**: 300
- **Batch size**: 16 (可根据GPU显存调整，但需记录)
- **Optimizer**: SGD
  - Momentum: 0.937
  - Weight decay: 0.0005
- **Learning rate**:
  - Initial LR: 0.01
  - Final LR: 0.0001
  - Warmup epochs: 3
  - Warmup momentum: 0.8
  - Warmup bias_lr: 0.1
- **Scheduler**: Cosine annealing
- **Loss weights**:
  - Box loss: 7.5
  - Cls loss: 0.5
  - DFL loss: 1.5

### 1.3 评估设置 (Evaluation Settings)
- **IoU 阈值 (NMS)**: 0.7
- **置信度阈值**: 0.001 (评估时), 0.25 (推理时)
- **最大检测数**: 300
- **验证频率**: 每个 epoch

### 1.4 硬件环境 (Hardware Environment)
所有实验必须记录以下信息：
- **GPU型号**: 例如 NVIDIA RTX 3090
- **GPU数量**: 1 (单卡训练)
- **CUDA版本**: 例如 11.8
- **显存容量**: 例如 24GB
- **Python版本**: 3.8+
- **PyTorch版本**: 2.0+
- **Ultralytics版本**: 记录准确的版本号或 Git commit hash

> 使用 `scripts/record_env.py` 自动记录环境信息。

---

## 2. 消融实验矩阵 (Ablation Study Matrix)

每个改进模块需要单独和组合测试，以验证其有效性和协同效果。

### 2.1 实验设计

| 实验ID | 实验名称 | Ghost模块 | ECA注意力 | P2检测头 | 描述 |
|--------|---------|-----------|-----------|----------|------|
| E0 | Baseline | ❌ | ❌ | ❌ | YOLOv8n 原始模型 |
| E1 | +Ghost | ✅ | ❌ | ❌ | 仅添加 GhostConv |
| E2 | +ECA | ❌ | ✅ | ❌ | 仅添加 ECA 注意力 |
| E3 | +P2 | ❌ | ❌ | ✅ | 仅添加 P2 小目标检测头 |
| E4 | Ghost+ECA | ✅ | ✅ | ❌ | GhostConv + ECA |
| E5 | Ghost+P2 | ✅ | ❌ | ✅ | GhostConv + P2 |
| E6 | ECA+P2 | ❌ | ✅ | ✅ | ECA + P2 |
| E7 | Full (All) | ✅ | ✅ | ✅ | 所有改进组合 |

### 2.2 命名规范
- **配置文件**: `experiments/ablation_E{id}_{name}.yaml`
- **日志目录**: `results/ablation/E{id}_{name}/`
- 示例: `experiments/ablation_E1_ghost.yaml` → `results/ablation/E1_ghost/`

---

## 3. 随机种子策略 (Random Seed Strategy)

### 3.1 多次运行
为了减少随机性影响，每个实验配置需要使用 **3个不同的随机种子** 进行训练：
```python
SEEDS = [0, 1, 2]
```

### 3.2 种子设置范围
使用 `scripts/set_determinism.py` 中的 `set_seed()` 函数，确保：
- Python `random` 模块
- NumPy `numpy.random`
- PyTorch `torch.manual_seed` (CPU & CUDA)
- cuDNN 确定性行为

### 3.3 结果报告格式
所有指标报告为 **均值 ± 标准差**：
```
mAP@0.5 = 0.756 ± 0.003
mAP@0.5:0.95 = 0.512 ± 0.005
```

### 3.4 实验记录
- **训练脚本**: 必须在命令行或配置中指定 `seed` 参数
- **日志分离**: 不同种子的结果保存在不同子目录
  ```
  results/ablation/E1_ghost/
  ├── seed_0/
  ├── seed_1/
  └── seed_2/
  ```

---

## 4. 评价指标 (Evaluation Metrics)

### 4.1 检测精度指标

#### 主要指标
- **mAP@0.5**: IoU=0.5 时的平均精度均值
- **mAP@0.5:0.95**: IoU 从 0.5 到 0.95 (步长 0.05) 的平均 mAP
- **Precision**: 精确率
- **Recall**: 召回率

#### 按尺寸分桶的 AP (Size-based AP)
根据 COCO 标准，将目标分为三类：
- **AP_small**: 面积 < 32² 像素
- **AP_medium**: 32² ≤ 面积 < 96² 像素
- **AP_large**: 面积 ≥ 96² 像素

> 对于小目标检测改进，特别关注 **AP_small** 的提升。

### 4.2 效率指标

#### 模型复杂度
- **Params (M)**: 模型参数量 (百万)
- **GFLOPs**: 浮点运算量 (十亿次)

#### 推理速度
在**固定硬件**上测量：
- **FPS** (Frames Per Second): 每秒处理帧数
- **Latency (ms)**: 单帧推理时间 (毫秒)
  - 测试条件: Batch size = 1, FP32, 640×640 输入
  - 测量方法: 预热 100 次，计时 300 次，取平均

### 4.3 定位精度指标

#### 中心点误差 (Center Point Error)
- **定义**: 预测框中心与真实框中心的欧式距离
- **单位**: 像素
- **计算**: 
  ```python
  error = sqrt((pred_cx - gt_cx)² + (pred_cy - gt_cy)²)
  ```
- **报告**: 所有正确检测 (IoU > 0.5) 的平均中心点误差

### 4.4 指标优先级
1. **主要关注**: mAP@0.5:0.95, mAP@0.5
2. **小目标性能**: AP_small
3. **效率平衡**: Params 和 GFLOPs 不应显著增加
4. **实时性**: FPS 应保持在合理范围 (>30 for 实时应用)

---

## 5. 结果输出文件命名规范与目录结构 (Output Structure)

### 5.1 目录结构
```
results/
├── metadata/
│   ├── env.json                    # 环境信息 (由 record_env.py 生成)
│   └── experiment_log.csv          # 所有实验的汇总表
├── ablation/
│   ├── E0_baseline/
│   │   ├── seed_0/
│   │   │   ├── weights/
│   │   │   │   ├── best.pt
│   │   │   │   └── last.pt
│   │   │   ├── results.csv
│   │   │   ├── results.png
│   │   │   ├── confusion_matrix.png
│   │   │   ├── F1_curve.png
│   │   │   ├── PR_curve.png
│   │   │   └── args.yaml
│   │   ├── seed_1/
│   │   │   └── ...
│   │   ├── seed_2/
│   │   │   └── ...
│   │   └── summary.json            # 3次运行的统计汇总
│   ├── E1_ghost/
│   │   └── ...
│   └── ...
├── inference/
│   └── test_images/                # 推理示例图片
└── analysis/
    ├── ablation_comparison.csv     # 消融实验对比表
    ├── ablation_comparison.png     # 可视化图表
    └── speed_comparison.csv        # 速度对比
```

### 5.2 文件命名规范

#### 配置文件 (experiments/)
```
experiments/ablation_E{实验ID}_{简称}.yaml
```
示例:
- `ablation_E0_baseline.yaml`
- `ablation_E1_ghost.yaml`
- `ablation_E7_full.yaml`

#### 训练结果 (results/)
```
results/ablation/E{实验ID}_{简称}/seed_{种子}/
```

#### 权重文件
- `best.pt`: 验证集上最优模型
- `last.pt`: 最后一个 epoch 的模型

#### 日志文件
- `results.csv`: 每个 epoch 的训练/验证指标
- `args.yaml`: 训练时使用的完整参数

### 5.3 汇总文件格式

#### summary.json (每个实验)
```json
{
  "experiment_id": "E1",
  "experiment_name": "ghost",
  "seeds": [0, 1, 2],
  "metrics": {
    "mAP50": {"mean": 0.756, "std": 0.003, "values": [0.754, 0.757, 0.757]},
    "mAP50-95": {"mean": 0.512, "std": 0.005, "values": [0.508, 0.514, 0.514]},
    "precision": {"mean": 0.712, "std": 0.004},
    "recall": {"mean": 0.689, "std": 0.006},
    "params_M": 3.01,
    "gflops": 8.1,
    "fps": 156.3,
    "center_error_px": 2.45
  }
}
```

#### experiment_log.csv (全局)
| exp_id | name | mAP50 | mAP50-95 | precision | recall | params_M | gflops | fps | train_time_h |
|--------|------|-------|----------|-----------|--------|----------|--------|-----|--------------|
| E0 | baseline | 0.745±0.002 | 0.501±0.004 | 0.705±0.003 | 0.681±0.005 | 3.15 | 8.2 | 161.2 | 4.5 |
| E1 | ghost | 0.756±0.003 | 0.512±0.005 | 0.712±0.004 | 0.689±0.006 | 3.01 | 8.1 | 156.3 | 4.3 |

---

## 6. 实验执行流程 (Execution Workflow)

### 6.1 准备阶段
1. 记录环境信息:
   ```bash
   python scripts/record_env.py
   ```

2. 准备实验配置:
   ```bash
   cp ultralytics/cfg/models/v8/yolov8.yaml experiments/ablation_E0_baseline.yaml
   # 根据消融矩阵修改配置文件
   ```

### 6.2 训练阶段
对于每个实验配置和每个种子：
```bash
for seed in 0 1 2; do
    python train.py \
        --cfg experiments/ablation_E1_ghost.yaml \
        --data datasets/data.yaml \
        --epochs 300 \
        --batch 16 \
        --seed $seed \
        --project results/ablation \
        --name E1_ghost/seed_$seed
done
```

### 6.3 评估阶段
```bash
python val.py \
    --weights results/ablation/E1_ghost/seed_0/weights/best.pt \
    --data datasets/data.yaml
```

### 6.4 汇总阶段
```bash
python scripts/aggregate_results.py --exp E1_ghost
```

---

## 7. 代码规范 (Coding Standards)

### 7.1 必须使用确定性设置
所有训练和评估脚本必须在开始时调用：
```python
from scripts.set_determinism import set_seed
set_seed(seed=args.seed)
```

### 7.2 日志记录
每次实验必须记录：
- 开始和结束时间
- 所有超参数
- 最终指标
- 异常情况

### 7.3 版本控制
- 每次实验前确保代码已提交到 Git
- 在 `summary.json` 中记录 commit hash

---

## 8. 质量检查清单 (Quality Checklist)

实验完成后，检查以下内容：

- [ ] 环境信息已记录到 `results/metadata/env.json`
- [ ] 每个实验运行了 3 次 (seeds: 0, 1, 2)
- [ ] 所有实验使用相同的数据划分
- [ ] 所有实验使用相同的超参数 (除了改进模块)
- [ ] 每个实验生成了 `summary.json`
- [ ] 指标以 `mean±std` 格式报告
- [ ] 训练曲线和混淆矩阵已保存
- [ ] 最优权重文件 `best.pt` 已保存
- [ ] 效率指标 (FPS, Params, GFLOPs) 已测量
- [ ] 所有文件遵循命名规范

---

## 9. 常见问题 (FAQ)

### Q1: 如果显存不足无法使用 batch_size=16 怎么办？
A: 可以减小 batch size，但必须在所有实验中保持一致，并在报告中明确说明。

### Q2: 训练时间过长怎么办？
A: 可以使用预训练权重或减少 epochs，但必须在所有对比实验中保持一致。

### Q3: 如何处理随机种子导致的训练失败？
A: 记录失败情况，使用备用种子 (3, 4, 5...)，但需在报告中说明。

### Q4: 是否可以使用多 GPU 训练？
A: 可以，但需要使用 DistributedDataParallel 并确保所有实验配置一致。

---

## 10. 参考文献

- [COCO Detection Evaluation](https://cocodataset.org/#detection-eval)
- [Papers with Code - Reproducibility](https://paperswithcode.com/rc2020)
- [PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)

---

**文档版本**: v1.0  
**最后更新**: 2026-02-02  
**维护者**: Ultralytics Research Team
