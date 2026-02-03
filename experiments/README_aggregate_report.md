# 结果汇总与报告生成使用指南

## 📋 概述

完成消融实验后，使用以下脚本自动生成汇总表格和可视化报告：

1. **`aggregate_results.py`**: 扫描实验结果，计算统计量，生成 CSV 汇总表
2. **`make_report.py`**: 生成 Markdown 报告和对比图表

## 🚀 快速使用

### 完整流程

```bash
# 1. 运行消融实验（需要先完成）
python scripts/run_ablation.py

# 2. 汇总结果（扫描 results/ 目录）
python scripts/aggregate_results.py

# 3. 生成报告（包含表格和图表）
python scripts/make_report.py
```

### 一键测试

使用模拟数据测试功能：

```bash
python scripts/test_aggregate_report.py
```

## 📊 aggregate_results.py - 结果汇总

### 功能

- 扫描 `results/runs/`、`results/evals/`、`results/benchmarks/` 目录
- 对每个实验计算跨种子的 **mean ± std**
- 生成 `results/summary/ablation_summary.csv` 和 `ablation_summary.json`
- 计算相对 baseline 的**相对提升百分比**

### 输出指标

#### 精度指标
- `mAP50`, `mAP50-95`: 主要精度指标
- `AP_small`, `AP_medium`, `AP_large`: 尺度分析
- `Precision`, `Recall`: 查准率和查全率

#### 定位指标
- `center_err_mean`: 中心点平均误差（像素）
- `center_err_median`: 中心点中位数误差
- `center_err_max`: 最大误差

#### 性能指标
- `fps`: 每秒帧数
- `latency_p95_ms`: 95分位延迟（毫秒）
- `params`: 参数量
- `gflops`: 浮点运算量
- `memory_peak_mb`: 峰值显存（MB）

### 使用示例

#### 基本用法

```bash
python scripts/aggregate_results.py
```

#### 指定结果目录

```bash
python scripts/aggregate_results.py \
    --results_dir results \
    --output_dir results/summary
```

#### 修改基线实验

```bash
python scripts/aggregate_results.py \
    --baseline my_baseline_exp
```

### 输出文件

#### 1. CSV 汇总表 (`ablation_summary.csv`)

包含所有实验的指标，格式示例：

```csv
experiment,mAP50_mean,mAP50_std,mAP50_rel,mAP50-95_mean,mAP50-95_std,mAP50-95_rel,...
baseline,0.751,0.003,0.0,0.549,0.000,0.0,...
ghost,0.729,0.006,-2.9,0.525,0.005,-4.3,...
eca,0.757,0.002,+0.9,0.549,0.011,+0.0,...
p2,0.762,0.005,+1.4,0.547,0.009,-0.2,...
```

字段说明：
- `{metric}_mean`: 均值
- `{metric}_std`: 标准差（样本标准差，ddof=1）
- `{metric}_rel`: 相对 baseline 的提升百分比

#### 2. JSON 完整数据 (`ablation_summary.json`)

包含所有统计量（mean, std, min, max, n）的完整数据，供报告生成使用。

### 命令行参数

```bash
python scripts/aggregate_results.py --help

选项：
  --results_dir TEXT    结果目录路径 [默认: results]
  --output_dir TEXT     输出目录 [默认: results/summary]
  --baseline TEXT       基线实验名称 [默认: baseline]
```

## 📈 make_report.py - 报告生成

### 功能

- 读取 `ablation_summary.json`
- 生成 Markdown 报告 (`ablation_report.md`)
- 生成三张对比图：
  - `plot_map_vs_fps.png`: mAP50-95 vs FPS
  - `plot_ap_small_vs_fps.png`: AP_small vs FPS  
  - `plot_center_err_vs_fps.png`: 中心误差 vs FPS
- 自动生成**不足分析模板**

### 使用示例

#### 基本用法

```bash
python scripts/make_report.py
```

#### 指定输入和输出

```bash
python scripts/make_report.py \
    --summary results/summary/ablation_summary.json \
    --output_dir results/summary \
    --protocol_file experiments/base_train.yaml
```

### 输出文件

#### 1. Markdown 报告 (`ablation_report.md`)

完整报告包含：

**1. 实验设置**
- 环境配置（OS、Python、PyTorch、CUDA、GPU）
- 实验协议（引用配置文件）
- 实验列表

**2. 主要结果**
- 完整指标对比表（Markdown 格式）
- 均值±标准差 + 相对提升百分比

**3. 可视化分析**
- 三张关键对比图及分析

**4. 关键发现**
- 最高精度模型
- 最快速度模型
- 最佳小目标检测模型

**5. 不足分析与未来改进方向**
- **自动分析每个实验**：
  - 精度变化（mAP50-95）
  - 小目标检测变化（AP_small）
  - 速度变化（FPS）
  - 参数量变化
  - 定位精度变化（center_err）
- **自动生成改进建议**：
  - 基于 trade-off 给出建议
  - 针对特定问题（如小目标下降）给出方向
- **总体结论**

**6. 附录**
- 数据来源
- 复现步骤

#### 2. 对比图表

##### plot_map_vs_fps.png
- X 轴：FPS（速度）
- Y 轴：mAP50-95（精度）
- 显示精度-速度 trade-off
- 标注帕累托前沿（Pareto Front）
- 带误差棒

##### plot_ap_small_vs_fps.png
- X 轴：FPS
- Y 轴：AP_small（小目标检测）
- 评估小目标检测改进效果

##### plot_center_err_vs_fps.png
- X 轴：FPS
- Y 轴：Center Error（中心点误差，像素）
- 误差越小越好
- 评估定位精度

### 命令行参数

```bash
python scripts/make_report.py --help

选项：
  --summary TEXT         汇总数据 JSON 文件 [默认: results/summary/ablation_summary.json]
  --output_dir TEXT      输出目录 [默认: results/summary]
  --protocol_file TEXT   实验协议文件 [默认: experiments/base_train.yaml]
```

## 📂 输出目录结构

完成后，`results/summary/` 目录包含：

```
results/summary/
├── ablation_summary.csv          # CSV 汇总表（可导入 Excel）
├── ablation_summary.json         # 完整统计数据（JSON）
├── ablation_report.md            # Markdown 报告
├── plot_map_vs_fps.png           # mAP50-95 vs FPS 图
├── plot_ap_small_vs_fps.png      # AP_small vs FPS 图
└── plot_center_err_vs_fps.png    # 中心误差 vs FPS 图
```

## 🔍 结果解读

### CSV 表格

每列含义：
- `experiment`: 实验名称
- `{metric}_mean`: 跨种子的平均值
- `{metric}_std`: 跨种子的标准差（衡量稳定性）
- `{metric}_rel`: 相对 baseline 的提升百分比

相对提升计算：
- **正指标**（越大越好，如 mAP、FPS）：`(current - baseline) / baseline * 100`
- **负指标**（越小越好，如 center_err、latency）：`(baseline - current) / baseline * 100`

### Markdown 报告

#### 主结果表

示例：

| 实验 | mAP50 | mAP50-95 | AP_small | FPS |
|------|-------|----------|----------|-----|
| baseline | 0.751±0.003 | 0.549±0.000 | 0.319±0.007 | 82.2±1.2 |
| ghost | 0.729±0.006 (-2.9%) | 0.525±0.005 (-4.3%) | 0.311±0.008 (-2.4%) | 102.0±0.5 (+24.1%) |

解读：
- ghost 精度下降 2.9%~4.3%
- 速度提升 24.1%
- 适合**实时性优先**场景

#### 自动分析

脚本会自动识别：
- ✅ 显著提升（> 0.5% 或 > 1.0%）
- ⚠️ 显著下降（< -0.5% 或 < -1.0%）
- 无显著变化

并给出针对性建议：
- 精度和速度双赢 → 深入分析成功因素
- 牺牲精度换速度 → 考虑知识蒸馏
- 牺牲速度换精度 → 考虑模型压缩
- 小目标下降 → 增强多尺度特征融合
- 定位精度下降 → 引入注意力机制

### 对比图表

#### 帕累托前沿（Pareto Front）

在 mAP vs FPS 图中，红色虚线连接帕累托最优点。

帕累托最优定义：
- 不存在其他点**同时**在精度和速度上更优

如何选择模型：
- **精度优先**：选择前沿上靠左的点（高精度，可接受的速度）
- **速度优先**：选择前沿上靠右的点（高速度，可接受的精度）
- **平衡**：选择前沿中部的点

## 🎓 最佳实践

### 1. 运行前检查

确保消融实验已完成：

```bash
# 检查训练输出
ls results/runs/*/seed*/weights/best.pt

# 检查评估输出
ls results/evals/*_test/metrics.json

# 检查基准测试
ls results/benchmarks/*_benchmark.json
```

### 2. 增量汇总

如果新增了实验，直接重新运行汇总即可：

```bash
python scripts/aggregate_results.py
python scripts/make_report.py
```

脚本会自动扫描所有实验并更新报告。

### 3. 自定义基线

如果你的基线实验不叫 "baseline"：

```bash
python scripts/aggregate_results.py --baseline my_baseline
```

### 4. 查看特定指标

从 CSV 提取特定列：

```bash
# 查看 mAP50-95 和 FPS
cat results/summary/ablation_summary.csv | cut -d',' -f1,5,26

# 使用 Python
import pandas as pd
df = pd.read_csv('results/summary/ablation_summary.csv')
print(df[['experiment', 'mAP50-95_mean', 'fps_mean']])
```

### 5. 导入 Excel

直接用 Excel 打开 `ablation_summary.csv`，或：

```python
import pandas as pd
df = pd.read_csv('results/summary/ablation_summary.csv')
df.to_excel('results/summary/ablation_summary.xlsx', index=False)
```

## 🐛 故障排查

### 问题 1: 未找到实验结果

**症状**: `❌ 未找到任何实验结果`

**解决方案**:
1. 确认消融实验已运行：`ls results/runs/`
2. 检查目录结构是否正确
3. 确认评估和基准测试已完成

### 问题 2: 基线实验不存在

**症状**: `⚠️ 基线实验 'baseline' 不存在，跳过相对提升计算`

**解决方案**:
- 方法 1: 确保有名为 "baseline" 的实验
- 方法 2: 指定正确的基线名称：`--baseline your_baseline`

### 问题 3: 某些实验缺少数据

**症状**: 扫描时显示 `✗ 评估目录不存在` 或 `✗ 基准测试不存在`

**解决方案**:
- 检查该实验是否完整运行
- 重新运行缺失的步骤：
  ```bash
  python scripts/run_eval_one.py --exp_name xxx --weights ... --split test
  python scripts/benchmark_model.py --weights ...
  ```

### 问题 4: 图表生成失败

**症状**: 图表文件不存在或损坏

**解决方案**:
1. 确认 matplotlib 已安装：`pip install matplotlib`
2. 检查数据是否有效（FPS > 0, 指标 > 0）
3. 查看错误日志

### 问题 5: Markdown 报告格式问题

**症状**: 表格显示不正确

**解决方案**:
- 使用支持 Markdown 的编辑器查看（如 VS Code、Typora）
- 或转换为 HTML：
  ```bash
  pip install markdown
  python -m markdown results/summary/ablation_report.md > report.html
  ```

## 📞 技术支持

**相关文件**:
- 汇总脚本: `scripts/aggregate_results.py`
- 报告生成: `scripts/make_report.py`
- 测试脚本: `scripts/test_aggregate_report.py`

**测试验证**:
```bash
# 使用模拟数据测试功能
python scripts/test_aggregate_report.py

# 预期输出: ✅ 所有测试通过
```

**示例输出**:
- CSV 表格: `results/summary/ablation_summary.csv`
- Markdown 报告: `results/summary/ablation_report.md`
- 对比图表: `results/summary/plot_*.png`

---

## 🎯 完整示例

假设你已完成 6 个实验（baseline, ghost, eca, p2, ghost_eca, ghost_eca_p2），每个 3 个种子：

```bash
# 步骤 1: 汇总结果
python scripts/aggregate_results.py
# 输出: ablation_summary.csv, ablation_summary.json

# 步骤 2: 生成报告
python scripts/make_report.py
# 输出: ablation_report.md, plot_*.png

# 步骤 3: 查看报告
cat results/summary/ablation_report.md

# 步骤 4: 查看图表（Windows）
start results/summary/plot_map_vs_fps.png

# 步骤 5: 导入 Excel 分析
# 用 Excel 打开 results/summary/ablation_summary.csv
```

生成的报告包含：
- ✅ 完整指标表（mean ± std + 相对提升）
- ✅ 三张对比图（精度、小目标、定位 vs FPS）
- ✅ 自动分析每个实验的优缺点
- ✅ 针对性改进建议

**报告可直接用于论文的实验章节！** 🎉

---

Happy analyzing! 📊
