"""
消融实验报告生成脚本

功能：
1. 读取 results/summary/ablation_summary.json
2. 生成 Markdown 报告 (results/summary/ablation_report.md)
3. 生成三张关键对比图（matplotlib）
4. 自动生成不足分析模板

报告内容：
1. 实验设置（引用协议和环境）
2. 主结果表（CSV 转 Markdown）
3. 可视化对比图：
   - mAP50-95 vs FPS
   - AP_small vs FPS
   - center_err vs FPS
4. 不足分析模板（基于数据自动生成）

使用方法：
    python scripts/make_report.py
    python scripts/make_report.py --summary results/summary/ablation_summary.json
"""

import sys
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime
import platform
import os

# Windows UTF-8 输出修复
if sys.platform == "win32":
    import codecs
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 设置中文字体（支持中文显示）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # Windows & Linux
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="消融实验报告生成")
    parser.add_argument(
        "--summary",
        type=str,
        default="results/summary/ablation_summary.json",
        help="汇总数据 JSON 文件"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/summary",
        help="输出目录"
    )
    parser.add_argument(
        "--protocol_file",
        type=str,
        default="experiments/base_train.yaml",
        help="实验协议文件"
    )
    return parser.parse_args()


def load_summary(summary_file):
    """加载汇总数据"""
    summary_file = Path(summary_file)
    
    if not summary_file.exists():
        print(f"❌ 汇总文件不存在: {summary_file}")
        print("\n请先运行:")
        print("  python scripts/aggregate_results.py")
        return None
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ 加载汇总数据: {len(data)} 个实验")
    return data


def get_system_info():
    """获取系统信息"""
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
    }
    
    try:
        import torch
        info.update({
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        })
    except ImportError:
        info.update({
            'pytorch_version': 'N/A',
            'cuda_available': False,
            'cuda_version': 'N/A',
            'gpu_name': 'N/A',
            'gpu_count': 0,
        })
    
    return info


def generate_markdown_table(data):
    """生成 Markdown 结果表"""
    lines = []
    
    # 表头
    lines.append("| 实验 | mAP50 | mAP50-95 | AP_small | AP_medium | AP_large | Center Err | FPS | Latency P95 | 参数量 | GFLOPs |")
    lines.append("|------|-------|----------|----------|-----------|----------|------------|-----|-------------|--------|--------|")
    
    # 按实验名排序
    for exp_name in sorted(data.keys()):
        stats = data[exp_name]
        
        # 提取指标
        map50 = stats.get('mAP50_mean', 0.0)
        map50_std = stats.get('mAP50_std', 0.0)
        map50_rel = stats.get('mAP50_rel', 0.0)
        
        map5095 = stats.get('mAP50-95_mean', 0.0)
        map5095_std = stats.get('mAP50-95_std', 0.0)
        map5095_rel = stats.get('mAP50-95_rel', 0.0)
        
        ap_small = stats.get('AP_small_mean', 0.0)
        ap_small_std = stats.get('AP_small_std', 0.0)
        ap_small_rel = stats.get('AP_small_rel', 0.0)
        
        ap_medium = stats.get('AP_medium_mean', 0.0)
        ap_medium_std = stats.get('AP_medium_std', 0.0)
        ap_medium_rel = stats.get('AP_medium_rel', 0.0)
        
        ap_large = stats.get('AP_large_mean', 0.0)
        ap_large_std = stats.get('AP_large_std', 0.0)
        ap_large_rel = stats.get('AP_large_rel', 0.0)
        
        center_err = stats.get('center_err_mean_mean', 0.0)
        center_err_std = stats.get('center_err_mean_std', 0.0)
        center_err_rel = stats.get('center_err_mean_rel', 0.0)
        
        fps = stats.get('fps_mean', 0.0)
        fps_std = stats.get('fps_std', 0.0)
        fps_rel = stats.get('fps_rel', 0.0)
        
        latency_p95 = stats.get('latency_p95_ms_mean', 0.0)
        latency_p95_std = stats.get('latency_p95_ms_std', 0.0)
        latency_p95_rel = stats.get('latency_p95_ms_rel', 0.0)
        
        params = stats.get('params_mean', 0.0)
        params_rel = stats.get('params_rel', 0.0)
        
        gflops = stats.get('gflops_mean', 0.0)
        gflops_rel = stats.get('gflops_rel', 0.0)
        
        # 格式化单元格（保留3位小数 + 相对提升）
        def fmt(val, std, rel):
            s = f"{val:.3f}±{std:.3f}"
            if abs(rel) > 0.01:  # 只显示 > 0.01% 的提升
                s += f" ({rel:+.1f}%)"
            return s
        
        def fmt_int(val, rel):
            s = f"{val/1e6:.2f}M" if val > 1e6 else f"{int(val)}"
            if abs(rel) > 0.01:
                s += f" ({rel:+.1f}%)"
            return s
        
        def fmt_float(val, rel):
            s = f"{val:.2f}"
            if abs(rel) > 0.01:
                s += f" ({rel:+.1f}%)"
            return s
        
        # 构建行
        row = [
            exp_name,
            fmt(map50, map50_std, map50_rel),
            fmt(map5095, map5095_std, map5095_rel),
            fmt(ap_small, ap_small_std, ap_small_rel),
            fmt(ap_medium, ap_medium_std, ap_medium_rel),
            fmt(ap_large, ap_large_std, ap_large_rel),
            fmt_float(center_err, center_err_rel) + "px",
            fmt_float(fps, fps_rel),
            fmt_float(latency_p95, latency_p95_rel) + "ms",
            fmt_int(params, params_rel),
            fmt_float(gflops, gflops_rel),
        ]
        
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)


def plot_metric_vs_fps(data, metric_key, metric_name, output_file, higher_is_better=True):
    """
    绘制 metric vs FPS 散点图
    
    Args:
        data: 汇总数据
        metric_key: 指标键（如 'mAP50-95_mean'）
        metric_name: 指标显示名称
        output_file: 输出文件路径
        higher_is_better: True 表示指标越大越好，False 表示越小越好
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 提取数据
    experiments = []
    fps_values = []
    metric_values = []
    fps_stds = []
    metric_stds = []
    
    for exp_name in sorted(data.keys()):
        stats = data[exp_name]
        
        fps = stats.get('fps_mean', 0.0)
        fps_std = stats.get('fps_std', 0.0)
        metric_val = stats.get(metric_key, 0.0)
        metric_std = stats.get(metric_key.replace('_mean', '_std'), 0.0)
        
        if fps > 0 and metric_val > 0:
            experiments.append(exp_name)
            fps_values.append(fps)
            metric_values.append(metric_val)
            fps_stds.append(fps_std)
            metric_stds.append(metric_std)
    
    if not experiments:
        print(f"⚠️  无有效数据，跳过图表: {output_file}")
        return
    
    # 绘制散点图（带误差棒）
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for i, exp_name in enumerate(experiments):
        ax.errorbar(
            fps_values[i],
            metric_values[i],
            xerr=fps_stds[i],
            yerr=metric_stds[i],
            fmt='o',
            markersize=10,
            label=exp_name,
            color=colors[i],
            capsize=5,
            alpha=0.7
        )
        
        # 添加标签
        ax.annotate(
            exp_name,
            (fps_values[i], metric_values[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
            alpha=0.8
        )
    
    # 标注帕累托前沿（如果 higher_is_better=True）
    if higher_is_better:
        # 找到帕累托最优点（右上角）
        pareto_indices = []
        for i in range(len(fps_values)):
            is_pareto = True
            for j in range(len(fps_values)):
                if i != j:
                    # 如果存在点 j 既更快又更准，则点 i 不是帕累托最优
                    if fps_values[j] >= fps_values[i] and metric_values[j] >= metric_values[i]:
                        if fps_values[j] > fps_values[i] or metric_values[j] > metric_values[i]:
                            is_pareto = False
                            break
            if is_pareto:
                pareto_indices.append(i)
        
        # 绘制帕累托前沿
        if len(pareto_indices) > 1:
            pareto_indices_sorted = sorted(pareto_indices, key=lambda i: fps_values[i])
            pareto_fps = [fps_values[i] for i in pareto_indices_sorted]
            pareto_metric = [metric_values[i] for i in pareto_indices_sorted]
            ax.plot(pareto_fps, pareto_metric, 'r--', alpha=0.3, linewidth=2, label='Pareto Front')
    
    # 设置标签和标题
    ax.set_xlabel('FPS (Frames Per Second)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} vs FPS Trade-off', fontsize=14, fontweight='bold')
    
    # 网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 图例
    ax.legend(loc='best', fontsize=9)
    
    # 保存
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 图表已保存: {output_file}")


def generate_plots(data, output_dir):
    """生成三张关键对比图"""
    output_dir = Path(output_dir)
    
    print("\n生成对比图...")
    
    # 1. mAP50-95 vs FPS
    plot_metric_vs_fps(
        data,
        'mAP50-95_mean',
        'mAP50-95',
        output_dir / 'plot_map_vs_fps.png',
        higher_is_better=True
    )
    
    # 2. AP_small vs FPS
    plot_metric_vs_fps(
        data,
        'AP_small_mean',
        'AP (Small Objects)',
        output_dir / 'plot_ap_small_vs_fps.png',
        higher_is_better=True
    )
    
    # 3. Center Error vs FPS (越小越好)
    plot_metric_vs_fps(
        data,
        'center_err_mean_mean',
        'Center Point Error (pixels)',
        output_dir / 'plot_center_err_vs_fps.png',
        higher_is_better=False
    )


def analyze_results(data, baseline='baseline'):
    """
    自动分析结果并生成不足分析模板
    
    返回 Markdown 格式的分析文本
    """
    lines = []
    
    if baseline not in data:
        lines.append("⚠️ 未找到基线实验，无法生成自动分析")
        return "\n".join(lines)
    
    baseline_stats = data[baseline]
    
    lines.append("## 5. 不足分析与未来改进方向")
    lines.append("")
    lines.append("### 5.1 自动分析结果")
    lines.append("")
    
    # 分析每个实验
    for exp_name in sorted(data.keys()):
        if exp_name == baseline:
            continue
        
        stats = data[exp_name]
        
        lines.append(f"#### {exp_name}")
        lines.append("")
        
        # 提取关键指标
        map5095_rel = stats.get('mAP50-95_rel', 0.0)
        ap_small_rel = stats.get('AP_small_rel', 0.0)
        ap_medium_rel = stats.get('AP_medium_rel', 0.0)
        ap_large_rel = stats.get('AP_large_rel', 0.0)
        fps_rel = stats.get('fps_rel', 0.0)
        params_rel = stats.get('params_rel', 0.0)
        center_err_rel = stats.get('center_err_mean_rel', 0.0)
        
        # 生成观察
        observations = []
        
        # 精度分析
        if map5095_rel > 0.5:
            observations.append(f"✅ **精度提升**: mAP50-95 相对 baseline 提升 {map5095_rel:.1f}%")
        elif map5095_rel < -0.5:
            observations.append(f"⚠️ **精度下降**: mAP50-95 相对 baseline 下降 {abs(map5095_rel):.1f}%")
        
        # 小目标分析
        if ap_small_rel > 1.0:
            observations.append(f"✅ **小目标检测增强**: AP_small 提升 {ap_small_rel:.1f}%")
        elif ap_small_rel < -1.0:
            observations.append(f"⚠️ **小目标检测下降**: AP_small 下降 {abs(ap_small_rel):.1f}%")
        
        # 速度分析
        if fps_rel > 5.0:
            observations.append(f"✅ **推理加速**: FPS 提升 {fps_rel:.1f}%")
        elif fps_rel < -5.0:
            observations.append(f"⚠️ **推理变慢**: FPS 下降 {abs(fps_rel):.1f}%")
        
        # 参数量分析
        if params_rel > 5.0:
            observations.append(f"⚠️ **参数量减少**: 参数量减少 {params_rel:.1f}%（注意：正值表示减少）")
        elif params_rel < -5.0:
            observations.append(f"⚠️ **参数量增加**: 参数量增加 {abs(params_rel):.1f}%")
        
        # 中心点误差分析
        if center_err_rel > 2.0:
            observations.append(f"✅ **定位精度提升**: 中心点误差减少 {center_err_rel:.1f}%")
        elif center_err_rel < -2.0:
            observations.append(f"⚠️ **定位精度下降**: 中心点误差增加 {abs(center_err_rel):.1f}%")
        
        # 输出观察
        if observations:
            for obs in observations:
                lines.append(f"- {obs}")
        else:
            lines.append("- 相对 baseline 无显著变化")
        
        lines.append("")
        
        # 生成建议
        lines.append("**改进建议**:")
        
        # 基于 trade-off 给建议
        if map5095_rel < 0 and fps_rel < 0:
            lines.append("- 当前改进既降低精度又降低速度，建议重新审视设计")
        elif map5095_rel < 0 and fps_rel > 0:
            lines.append("- 牺牲精度换取速度，可考虑添加知识蒸馏恢复精度")
        elif map5095_rel > 0 and fps_rel < 0:
            lines.append("- 牺牲速度换取精度，可考虑模型压缩技术（剪枝、量化）")
        else:
            lines.append("- 当前改进达到精度和速度双赢，建议深入分析成功因素")
        
        # 小目标建议
        if ap_small_rel < -2.0:
            lines.append("- 小目标检测下降明显，建议增强多尺度特征融合或增加 P2 层")
        
        # 定位建议
        if center_err_rel < -2.0:
            lines.append("- 定位精度下降，建议引入注意力机制或改进损失函数")
        
        lines.append("")
    
    lines.append("### 5.2 总体结论")
    lines.append("")
    lines.append("基于以上自动分析，实验团队应：")
    lines.append("")
    lines.append("1. **优先选择**：在精度-速度 trade-off 曲线上位于帕累托前沿的模型")
    lines.append("2. **深入分析**：对精度或速度显著下降的实验，检查实现细节和超参数")
    lines.append("3. **组合策略**：考虑将多个有效改进组合，但需注意兼容性")
    lines.append("4. **应用场景**：根据实际需求（实时性 vs 精度）选择最适合的模型")
    lines.append("")
    
    return "\n".join(lines)


def generate_report(data, output_file, protocol_file=None):
    """生成完整的 Markdown 报告"""
    lines = []
    
    # 标题
    lines.append("# 消融实验报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # 1. 实验设置
    lines.append("## 1. 实验设置")
    lines.append("")
    
    # 1.1 环境配置
    lines.append("### 1.1 环境配置")
    lines.append("")
    system_info = get_system_info()
    lines.append(f"- **操作系统**: {system_info['platform']}")
    lines.append(f"- **Python 版本**: {system_info['python_version']}")
    lines.append(f"- **PyTorch 版本**: {system_info['pytorch_version']}")
    lines.append(f"- **CUDA 版本**: {system_info['cuda_version']}")
    lines.append(f"- **GPU**: {system_info['gpu_name']} × {system_info['gpu_count']}")
    lines.append("")
    
    # 1.2 实验协议
    lines.append("### 1.2 实验协议")
    lines.append("")
    
    if protocol_file and Path(protocol_file).exists():
        lines.append(f"详见配置文件: `{protocol_file}`")
        lines.append("")
        
        # 读取协议文件的关键参数
        try:
            import yaml
            with open(protocol_file, 'r', encoding='utf-8') as f:
                protocol = yaml.safe_load(f)
            
            lines.append("**关键超参数**:")
            lines.append("")
            
            if isinstance(protocol, dict):
                for key in ['epochs', 'batch', 'imgsz', 'lr0', 'weight_decay', 'optimizer']:
                    if key in protocol:
                        lines.append(f"- `{key}`: {protocol[key]}")
            lines.append("")
        except:
            pass
    else:
        lines.append("（协议文件未找到，请手动补充实验设置）")
        lines.append("")
    
    # 1.3 实验列表
    lines.append("### 1.3 实验列表")
    lines.append("")
    lines.append(f"本次消融实验共包含 **{len(data)}** 个配置，每个配置使用 **3 个随机种子** 重复训练。")
    lines.append("")
    
    experiment_descriptions = {
        'baseline': 'Baseline model (标准 YOLOv8n 架构)',
        'ghost': 'Ghost 卷积（参数量减少 ~32%）',
        'eca': 'ECA 注意力机制',
        'p2': 'P2 层（增强小目标检测）',
        'ghost_eca': 'Ghost + ECA 组合',
        'ghost_eca_p2': '完整改进模型（Ghost + ECA + P2）',
    }
    
    for exp_name in sorted(data.keys()):
        desc = experiment_descriptions.get(exp_name, 'N/A')
        lines.append(f"- **{exp_name}**: {desc}")
    
    lines.append("")
    
    # 2. 主结果表
    lines.append("## 2. 主要结果")
    lines.append("")
    lines.append("### 2.1 完整指标对比")
    lines.append("")
    lines.append("下表展示了所有实验的平均指标（mean ± std）及相对 baseline 的提升百分比。")
    lines.append("")
    
    table = generate_markdown_table(data)
    lines.append(table)
    lines.append("")
    
    lines.append("**说明**:")
    lines.append("")
    lines.append("- 数值格式: `均值±标准差 (相对提升%)`")
    lines.append("- 正值表示相对 baseline 提升，负值表示下降")
    lines.append("- 对于 Center Err、Latency、参数量等指标，正值表示**减少**（改进）")
    lines.append("")
    
    # 3. 可视化分析
    lines.append("## 3. 可视化分析")
    lines.append("")
    
    lines.append("### 3.1 mAP50-95 vs FPS Trade-off")
    lines.append("")
    lines.append("![mAP50-95 vs FPS](plot_map_vs_fps.png)")
    lines.append("")
    lines.append("**分析**: 该图展示了不同模型在精度（mAP50-95）和速度（FPS）之间的权衡。位于右上角的模型达到最佳平衡。")
    lines.append("")
    
    lines.append("### 3.2 小目标检测性能 vs FPS")
    lines.append("")
    lines.append("![AP_small vs FPS](plot_ap_small_vs_fps.png)")
    lines.append("")
    lines.append("**分析**: 该图聚焦于小目标检测能力。P2 层等改进应在此图中显示显著优势。")
    lines.append("")
    
    lines.append("### 3.3 定位精度 vs FPS")
    lines.append("")
    lines.append("![Center Error vs FPS](plot_center_err_vs_fps.png)")
    lines.append("")
    lines.append("**分析**: 该图评估中心点定位精度。误差越小表示定位越准确。")
    lines.append("")
    
    # 4. 关键发现
    lines.append("## 4. 关键发现")
    lines.append("")
    
    # 找到最佳模型
    best_map = max(data.items(), key=lambda x: x[1].get('mAP50-95_mean', 0.0))
    best_fps = max(data.items(), key=lambda x: x[1].get('fps_mean', 0.0))
    best_small = max(data.items(), key=lambda x: x[1].get('AP_small_mean', 0.0))
    
    lines.append(f"- **最高精度**: {best_map[0]} (mAP50-95 = {best_map[1].get('mAP50-95_mean', 0.0):.3f})")
    lines.append(f"- **最快速度**: {best_fps[0]} (FPS = {best_fps[1].get('fps_mean', 0.0):.1f})")
    lines.append(f"- **最佳小目标**: {best_small[0]} (AP_small = {best_small[1].get('AP_small_mean', 0.0):.3f})")
    lines.append("")
    
    # 5. 不足分析
    analysis = analyze_results(data, baseline='baseline')
    lines.append(analysis)
    
    # 6. 附录
    lines.append("## 6. 附录")
    lines.append("")
    lines.append("### 6.1 数据来源")
    lines.append("")
    lines.append("- 训练输出: `results/runs/`")
    lines.append("- 评估输出: `results/evals/`")
    lines.append("- 基准测试: `results/benchmarks/`")
    lines.append("- 汇总数据: `results/summary/ablation_summary.json`")
    lines.append("")
    
    lines.append("### 6.2 复现步骤")
    lines.append("")
    lines.append("```bash")
    lines.append("# 1. 运行消融实验")
    lines.append("python scripts/run_ablation.py")
    lines.append("")
    lines.append("# 2. 汇总结果")
    lines.append("python scripts/aggregate_results.py")
    lines.append("")
    lines.append("# 3. 生成报告")
    lines.append("python scripts/make_report.py")
    lines.append("```")
    lines.append("")
    
    # 写入文件
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"\n✓ 报告已保存: {output_file}")


def main():
    """主函数"""
    args = parse_args()
    
    print("\n" + "="*80)
    print("消融实验报告生成")
    print("="*80)
    print(f"汇总数据: {args.summary}")
    print(f"输出目录: {args.output_dir}")
    
    # 1. 加载汇总数据
    print("\n" + "="*80)
    print("步骤 1: 加载汇总数据")
    print("="*80)
    data = load_summary(args.summary)
    
    if not data:
        return 1
    
    # 2. 生成对比图
    print("\n" + "="*80)
    print("步骤 2: 生成对比图")
    print("="*80)
    generate_plots(data, args.output_dir)
    
    # 3. 生成 Markdown 报告
    print("\n" + "="*80)
    print("步骤 3: 生成 Markdown 报告")
    print("="*80)
    output_md = Path(args.output_dir) / "ablation_report.md"
    generate_report(data, output_md, args.protocol_file)
    
    print("\n" + "="*80)
    print("✅ 报告生成完成！")
    print("="*80)
    print(f"\n输出文件:")
    print(f"  - 报告: {output_md}")
    print(f"  - 图表 1: {Path(args.output_dir) / 'plot_map_vs_fps.png'}")
    print(f"  - 图表 2: {Path(args.output_dir) / 'plot_ap_small_vs_fps.png'}")
    print(f"  - 图表 3: {Path(args.output_dir) / 'plot_center_err_vs_fps.png'}")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
