# 消融实验模型配置文档

## 概述

本文档记录了为 Ultralytics YOLOv8 实现的消融实验模型配置，确保每个实验只改变一个因素，满足"消融可控"的要求。

## 📦 实现的模块

### 1. GhostConv & GhostBottleneck (已有)
- **位置**: `ultralytics/nn/modules/conv.py`, `ultralytics/nn/modules/block.py`
- **状态**: ✅ 已存在于代码库中
- **功能**: Ghost 卷积模块，使用更少的参数生成更多特征
- **参考**: [GhostNet 论文](https://arxiv.org/abs/1911.11907)

### 2. ECA (Efficient Channel Attention) (新增)
- **位置**: `ultralytics/nn/modules/conv.py`
- **状态**: ✅ 新实现
- **功能**: 轻量级通道注意力机制，使用 1D 卷积实现跨通道交互
- **优势**: 比 SE 模块更轻量（无全连接层），参数量极少
- **参考**: [ECA-Net 论文](https://arxiv.org/abs/1910.03151)

## 🔬 消融实验配置

### 模型配置汇总

| 实验 | 配置文件 | 改变点 | 参数量 | 相对 Baseline |
|------|----------|--------|--------|---------------|
| E0: Baseline | `yolov8n_baseline.yaml` | 无（标准 YOLOv8n） | 3,157,200 | - |
| E1: Ghost | `yolov8n_ghost.yaml` | C2f → C3Ghost | 2,144,916 | -32.06% |
| E2: ECA | `yolov8n_eca.yaml` | 插入 ECA 模块 | 3,157,212 | +0.00% |
| E3: P2 Head | `yolov8n_p2.yaml` | 增加 P2 检测头 | 3,354,144 | +6.24% |
| E4: Ghost+ECA | `yolov8n_ghost_eca.yaml` | E1 + E2 | 2,144,928 | -32.06% |
| E7: Full | `yolov8n_ghost_eca_p2.yaml` | E1 + E2 + E3 | 2,317,764 | -26.59% |

### 详细说明

#### E0: Baseline (yolov8n_baseline.yaml)
- **唯一改变点**: 无（标准 YOLOv8n 结构）
- **用途**: 作为所有消融实验的参考基线
- **重要**: 此文件一旦确定后不得修改

#### E1: Ghost (yolov8n_ghost.yaml)
- **唯一改变点**: 将 backbone 和 head 中的 C2f 替换为 C3Ghost
- **改动位置**:
  - Backbone: 第 2, 4, 6, 8 层
  - Head: 第 12, 15, 18, 21 层
- **效果**: 参数量减少 32%

#### E2: ECA (yolov8n_eca.yaml)
- **唯一改变点**: 在每个 stage 的 C2f 后插入 ECA 注意力模块
- **改动位置**: Backbone 第 2, 4, 6, 8 层后各插入一个 ECA
- **效果**: 参数量几乎不变（仅增加 12 个参数）
- **注意**: 插入后层号发生变化，需调整 Concat 索引

#### E3: P2 Head (yolov8n_p2.yaml)
- **唯一改变点**: 增加 P2 (P2/4) 检测头，用于小目标检测
- **改动位置**: Head 增加 P2 上采样和检测分支
- **检测尺度**: P2/4, P3/8, P4/16, P5/32 (4 个)
- **效果**: 参数量增加 6.24%

#### E4: Ghost+ECA (yolov8n_ghost_eca.yaml)
- **唯一改变点**: 组合 Ghost 轻量化和 ECA 注意力
- **组合**: E1 + E2
- **效果**: 参数量减少 32%，同时引入注意力机制

#### E7: Full (yolov8n_ghost_eca_p2.yaml)
- **唯一改变点**: 组合所有改进
- **组合**: E1 + E2 + E3
- **效果**: 参数量减少 26.59%，同时具备轻量化、注意力、小目标检测

## 🧪 测试与验证

### 模块单元测试

```bash
# 运行模块单元测试
uv run python scripts/test_modules.py

# 显示详细信息
uv run python scripts/test_modules.py --verbose
```

**测试内容**:
- GhostConv 前向传播和梯度反向传播
- GhostBottleneck (stride=1 和 stride=2)
- C3Ghost 模块
- ECA 模块 (kernel_size=3,5,7)
- Conv + ECA 组合

### 模型配置验证

```bash
# 验证所有模型配置
uv run python scripts/validate_models.py

# 验证单个模型
uv run python scripts/validate_models.py --model yolov8n_baseline.yaml

# 显示详细信息
uv run python scripts/validate_models.py --verbose
```

**验证内容**:
- 模型加载成功
- 前向传播测试
- 参数量统计
- 参数量对比分析

### 验证结果

✅ **所有测试通过**:
- 5/5 模块单元测试通过
- 6/6 模型配置验证通过

## 📝 使用示例

### 训练示例

```bash
# E0: Baseline
uv run python -m ultralytics train \
    model=ultralytics/cfg/models/v8/yolov8n_baseline.yaml \
    data=datasets/openparts/data.yaml \
    epochs=100 batch=8 \
    name=E0_baseline_seed0

# E1: Ghost
uv run python -m ultralytics train \
    model=ultralytics/cfg/models/v8/yolov8n_ghost.yaml \
    data=datasets/openparts/data.yaml \
    epochs=100 batch=8 \
    name=E1_ghost_seed0

# E2: ECA
uv run python -m ultralytics train \
    model=ultralytics/cfg/models/v8/yolov8n_eca.yaml \
    data=datasets/openparts/data.yaml \
    epochs=100 batch=8 \
    name=E2_eca_seed0
```

### 评估示例

```bash
# 评估 Baseline
uv run python -m ultralytics val \
    model=results/train/E0_baseline_seed0/weights/best.pt \
    data=datasets/openparts/data.yaml \
    split=val

# 评估 Ghost
uv run python -m ultralytics val \
    model=results/train/E1_ghost_seed0/weights/best.pt \
    data=datasets/openparts/data.yaml \
    split=test
```

### Python API 使用

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("ultralytics/cfg/models/v8/yolov8n_baseline.yaml")

# 训练
results = model.train(
    data="datasets/openparts/data.yaml",
    epochs=100,
    batch=8,
    name="E0_baseline_seed0"
)

# 评估
metrics = model.val(split="val")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

## 🔍 关键设计原则

### 1. 单因素变量控制
每个配置文件只改变一个因素，确保消融实验的有效性：
- E1: 只改变模块类型（C2f → C3Ghost）
- E2: 只增加注意力机制
- E3: 只增加检测头
- E4, E7: 明确组合已验证的改进

### 2. 最小化差异
- 所有配置都基于相同的 Baseline
- 保持相同的 backbone 结构
- 使用相同的超参数配置
- 仅修改必要的部分

### 3. 可追溯性
- YAML 文件头部明确标注"唯一改变点"
- 在改动的行后添加注释标记（← 改动）
- 记录调整的索引位置

### 4. 可复现性
- 所有配置文件冻结后不再修改
- 使用相同的训练配置（experiments/base_train.yaml）
- 使用相同的评估配置（experiments/base_eval.yaml）
- 多种子实验（seed=0,1,2）

## 📂 文件结构

```
ultralytics/
├── cfg/
│   └── models/
│       └── v8/
│           ├── yolov8n_baseline.yaml      # E0: Baseline
│           ├── yolov8n_ghost.yaml         # E1: Ghost
│           ├── yolov8n_eca.yaml           # E2: ECA
│           ├── yolov8n_p2.yaml            # E3: P2 Head
│           ├── yolov8n_ghost_eca.yaml     # E4: Ghost+ECA
│           └── yolov8n_ghost_eca_p2.yaml  # E7: Full
├── nn/
│   └── modules/
│       ├── conv.py          # GhostConv, ECA (新增)
│       ├── block.py         # GhostBottleneck, C3Ghost
│       └── __init__.py      # 模块导出
└── scripts/
    ├── test_modules.py      # 模块单元测试
    └── validate_models.py   # 模型配置验证
```

## ⚠️ 注意事项

1. **Baseline 冻结**: `yolov8n_baseline.yaml` 确定后不得修改
2. **索引调整**: 插入新层后需调整 Concat 的 from 索引
3. **通道数匹配**: ECA 的 channels 参数必须与前一层输出通道数一致
4. **检测头数量**: P2 模型使用 4 个检测头，其他使用 3 个
5. **参数统计**: 使用 `validate_models.py` 验证参数量变化

## 🎯 下一步

1. 使用 `experiments/base_train.yaml` 训练所有模型
2. 每个模型运行 3 个种子（0, 1, 2）
3. 使用 `experiments/base_eval.yaml` 在验证集和测试集上评估
4. 记录所有实验结果到 `results/` 目录
5. 生成消融实验对比表格和可视化

## 📚 参考文献

1. **YOLOv8**: Ultralytics YOLOv8 - https://github.com/ultralytics/ultralytics
2. **GhostNet**: Han et al., "GhostNet: More Features from Cheap Operations", CVPR 2020
3. **ECA-Net**: Wang et al., "ECA-Net: Efficient Channel Attention for Deep CNNs", CVPR 2020
4. **P2 Detection**: 多尺度检测头，用于小目标检测

---

**文档版本**: 1.0  
**最后更新**: 2026-02-02  
**维护者**: Ultralytics Ablation Experiments
