"""
消融实验结果汇总脚本

功能：
1. 扫描 results/runs、results/evals、results/benchmarks 目录
2. 对每个实验计算跨种子的 mean±std
3. 生成 results/summary/ablation_summary.csv
4. 计算相对 baseline 的提升

输出指标：
- mAP50, mAP50-95, AP_small, AP_medium, AP_large
- center_err_mean
- fps, latency_p95
- params, gflops

使用方法：
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --results_dir results
"""

import sys
from pathlib import Path
import argparse
import json
import csv
import numpy as np
from collections import defaultdict
import os

# Windows UTF-8 输出修复
if sys.platform == "win32":
    import codecs
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="消融实验结果汇总")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="结果目录路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/summary",
        help="输出目录"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="baseline",
        help="基线实验名称"
    )
    return parser.parse_args()


def load_eval_metrics(eval_dir):
    """加载评估指标"""
    metrics = {}
    
    # 加载标准指标
    metrics_file = eval_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            metrics.update({
                'mAP50': data.get('mAP50', 0.0),
                'mAP50-95': data.get('mAP50-95', 0.0),
                'Precision': data.get('Precision', 0.0),
                'Recall': data.get('Recall', 0.0),
            })
    
    # 加载尺度指标
    size_file = eval_dir / "size_metrics.json"
    if size_file.exists():
        with open(size_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            metrics.update({
                'AP_small': data.get('AP_small', 0.0),
                'AP_medium': data.get('AP_medium', 0.0),
                'AP_large': data.get('AP_large', 0.0),
            })
    
    # 加载中心点误差
    center_file = eval_dir / "center_errors.json"
    if center_file.exists():
        with open(center_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            metrics.update({
                'center_err_mean': data.get('mean_error_pixels', 0.0),
                'center_err_median': data.get('median_error_pixels', 0.0),
                'center_err_max': data.get('max_error_pixels', 0.0),
            })
    
    return metrics


def load_benchmark_metrics(bench_file):
    """加载基准测试指标"""
    metrics = {}
    
    if bench_file.exists():
        with open(bench_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 提取关键指标
            metrics.update({
                'params': data.get('model_params', 0),
                'gflops': data.get('model_gflops', 0.0),
                'latency_mean_ms': data.get('latency_mean_ms', 0.0),
                'latency_std_ms': data.get('latency_std_ms', 0.0),
                'latency_p50_ms': data.get('latency_p50_ms', 0.0),
                'latency_p95_ms': data.get('latency_p95_ms', 0.0),
                'latency_p99_ms': data.get('latency_p99_ms', 0.0),
                'fps': data.get('fps', 0.0),
                'memory_allocated_mb': data.get('memory_allocated_mb', 0.0),
                'memory_peak_mb': data.get('memory_peak_mb', 0.0),
            })
    
    return metrics


def scan_results(results_dir):
    """
    扫描结果目录并收集数据
    
    返回结构：
    {
        'baseline': {
            'seed0': {'mAP50': 0.75, 'fps': 80, ...},
            'seed1': {...},
            'seed2': {...}
        },
        'ghost': {...},
        ...
    }
    """
    results_dir = Path(results_dir)
    data = defaultdict(dict)
    
    # 扫描 results/runs 目录
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        print(f"⚠️  运行目录不存在: {runs_dir}")
        return data
    
    # 遍历每个实验
    for exp_dir in runs_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        print(f"\n扫描实验: {exp_name}")
        
        # 遍历每个种子
        for seed_dir in exp_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith('seed'):
                continue
            
            seed_name = seed_dir.name  # e.g., 'seed0'
            seed_num = seed_name.replace('seed', '')
            
            print(f"  - {seed_name}")
            
            # 初始化该种子的指标字典
            seed_metrics = {}
            
            # 1. 加载评估指标 (test split)
            eval_dir = results_dir / "evals" / f"{exp_name}_{seed_name}_test"
            if eval_dir.exists():
                eval_metrics = load_eval_metrics(eval_dir)
                seed_metrics.update(eval_metrics)
                print(f"    ✓ 评估指标: {len(eval_metrics)} 项")
            else:
                print(f"    ✗ 评估目录不存在: {eval_dir}")
            
            # 2. 加载基准测试指标
            bench_file = results_dir / "benchmarks" / f"{exp_name}_{seed_name}_benchmark.json"
            if bench_file.exists():
                bench_metrics = load_benchmark_metrics(bench_file)
                seed_metrics.update(bench_metrics)
                print(f"    ✓ 基准测试: {len(bench_metrics)} 项")
            else:
                print(f"    ✗ 基准测试不存在: {bench_file}")
            
            # 保存该种子的完整指标
            if seed_metrics:
                data[exp_name][seed_name] = seed_metrics
    
    return data


def aggregate_metrics(data):
    """
    对每个实验计算跨种子的统计量 (mean ± std)
    
    返回结构：
    {
        'baseline': {
            'mAP50_mean': 0.75,
            'mAP50_std': 0.02,
            'fps_mean': 80,
            'fps_std': 1.5,
            ...
        },
        'ghost': {...},
        ...
    }
    """
    aggregated = {}
    
    # 定义需要聚合的指标
    metrics_to_aggregate = [
        'mAP50', 'mAP50-95', 'Precision', 'Recall',
        'AP_small', 'AP_medium', 'AP_large',
        'center_err_mean', 'center_err_median', 'center_err_max',
        'params', 'gflops',
        'latency_mean_ms', 'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms',
        'fps',
        'memory_allocated_mb', 'memory_peak_mb',
    ]
    
    for exp_name, seeds_data in data.items():
        print(f"\n聚合实验: {exp_name} ({len(seeds_data)} 个种子)")
        
        exp_stats = {}
        
        # 收集所有种子的数据
        for metric in metrics_to_aggregate:
            values = []
            for seed_name, seed_metrics in seeds_data.items():
                if metric in seed_metrics:
                    values.append(seed_metrics[metric])
            
            if values:
                values = np.array(values)
                exp_stats[f'{metric}_mean'] = float(np.mean(values))
                exp_stats[f'{metric}_std'] = float(np.std(values, ddof=1) if len(values) > 1 else 0.0)
                exp_stats[f'{metric}_min'] = float(np.min(values))
                exp_stats[f'{metric}_max'] = float(np.max(values))
                exp_stats[f'{metric}_n'] = len(values)
                
                print(f"  {metric}: {exp_stats[f'{metric}_mean']:.4f} ± {exp_stats[f'{metric}_std']:.4f}")
        
        aggregated[exp_name] = exp_stats
    
    return aggregated


def calculate_relative_improvements(aggregated, baseline_name='baseline'):
    """
    计算相对于 baseline 的提升
    
    添加字段：
    - {metric}_rel: 相对提升百分比
    """
    if baseline_name not in aggregated:
        print(f"⚠️  基线实验 '{baseline_name}' 不存在，跳过相对提升计算")
        return aggregated
    
    baseline = aggregated[baseline_name]
    
    # 定义需要计算相对提升的指标（越大越好）
    positive_metrics = [
        'mAP50', 'mAP50-95', 'Precision', 'Recall',
        'AP_small', 'AP_medium', 'AP_large',
        'fps',
    ]
    
    # 定义需要计算相对降低的指标（越小越好）
    negative_metrics = [
        'center_err_mean', 'center_err_median', 'center_err_max',
        'latency_mean_ms', 'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms',
        'params', 'gflops',
        'memory_allocated_mb', 'memory_peak_mb',
    ]
    
    for exp_name, exp_stats in aggregated.items():
        print(f"\n计算相对提升: {exp_name}")
        
        # 对于 positive metrics（越大越好）
        for metric in positive_metrics:
            mean_key = f'{metric}_mean'
            if mean_key in exp_stats and mean_key in baseline:
                baseline_val = baseline[mean_key]
                current_val = exp_stats[mean_key]
                
                if baseline_val > 0:
                    rel_improvement = ((current_val - baseline_val) / baseline_val) * 100
                    exp_stats[f'{metric}_rel'] = rel_improvement
                    
                    if exp_name != baseline_name:
                        sign = '+' if rel_improvement > 0 else ''
                        print(f"  {metric}: {sign}{rel_improvement:.2f}%")
        
        # 对于 negative metrics（越小越好）
        for metric in negative_metrics:
            mean_key = f'{metric}_mean'
            if mean_key in exp_stats and mean_key in baseline:
                baseline_val = baseline[mean_key]
                current_val = exp_stats[mean_key]
                
                if baseline_val > 0:
                    # 注意：这里取负号，使得减少是正面的
                    rel_improvement = ((baseline_val - current_val) / baseline_val) * 100
                    exp_stats[f'{metric}_rel'] = rel_improvement
                    
                    if exp_name != baseline_name:
                        sign = '+' if rel_improvement > 0 else ''
                        print(f"  {metric}: {sign}{rel_improvement:.2f}%")
    
    return aggregated


def save_summary_csv(aggregated, output_file):
    """保存汇总结果到 CSV"""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 定义输出列（按优先级排序）
    columns = [
        'experiment',
        'mAP50_mean', 'mAP50_std', 'mAP50_rel',
        'mAP50-95_mean', 'mAP50-95_std', 'mAP50-95_rel',
        'AP_small_mean', 'AP_small_std', 'AP_small_rel',
        'AP_medium_mean', 'AP_medium_std', 'AP_medium_rel',
        'AP_large_mean', 'AP_large_std', 'AP_large_rel',
        'Precision_mean', 'Precision_std', 'Precision_rel',
        'Recall_mean', 'Recall_std', 'Recall_rel',
        'center_err_mean_mean', 'center_err_mean_std', 'center_err_mean_rel',
        'fps_mean', 'fps_std', 'fps_rel',
        'latency_p95_ms_mean', 'latency_p95_ms_std', 'latency_p95_ms_rel',
        'params_mean', 'params_std', 'params_rel',
        'gflops_mean', 'gflops_std', 'gflops_rel',
        'memory_peak_mb_mean', 'memory_peak_mb_std', 'memory_peak_mb_rel',
    ]
    
    # 写入 CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        
        for exp_name in sorted(aggregated.keys()):
            row = {'experiment': exp_name}
            row.update(aggregated[exp_name])
            writer.writerow(row)
    
    print(f"\n✓ 汇总结果已保存: {output_file}")
    print(f"  实验数: {len(aggregated)}")
    print(f"  指标数: {len(columns) - 1}")


def print_summary_table(aggregated):
    """打印简要汇总表"""
    print("\n" + "="*100)
    print("消融实验结果汇总")
    print("="*100)
    
    # 表头
    header = f"{'实验':<20} {'mAP50':<12} {'mAP50-95':<12} {'AP_small':<12} {'FPS':<10} {'参数量':<12}"
    print(header)
    print("-"*100)
    
    # 按实验名排序
    for exp_name in sorted(aggregated.keys()):
        stats = aggregated[exp_name]
        
        map50 = stats.get('mAP50_mean', 0.0)
        map50_std = stats.get('mAP50_std', 0.0)
        map50_rel = stats.get('mAP50_rel', 0.0)
        
        map5095 = stats.get('mAP50-95_mean', 0.0)
        map5095_std = stats.get('mAP50-95_std', 0.0)
        map5095_rel = stats.get('mAP50-95_rel', 0.0)
        
        ap_small = stats.get('AP_small_mean', 0.0)
        ap_small_std = stats.get('AP_small_std', 0.0)
        ap_small_rel = stats.get('AP_small_rel', 0.0)
        
        fps = stats.get('fps_mean', 0.0)
        fps_std = stats.get('fps_std', 0.0)
        fps_rel = stats.get('fps_rel', 0.0)
        
        params = stats.get('params_mean', 0.0)
        params_rel = stats.get('params_rel', 0.0)
        
        # 格式化输出
        map50_str = f"{map50:.3f}±{map50_std:.3f}"
        if map50_rel != 0:
            map50_str += f"({map50_rel:+.1f}%)"
        
        map5095_str = f"{map5095:.3f}±{map5095_std:.3f}"
        if map5095_rel != 0:
            map5095_str += f"({map5095_rel:+.1f}%)"
        
        ap_small_str = f"{ap_small:.3f}±{ap_small_std:.3f}"
        if ap_small_rel != 0:
            ap_small_str += f"({ap_small_rel:+.1f}%)"
        
        fps_str = f"{fps:.1f}±{fps_std:.1f}"
        if fps_rel != 0:
            fps_str += f"({fps_rel:+.1f}%)"
        
        params_m = params / 1e6
        params_str = f"{params_m:.2f}M"
        if params_rel != 0:
            params_str += f"({params_rel:+.1f}%)"
        
        row = f"{exp_name:<20} {map50_str:<12} {map5095_str:<12} {ap_small_str:<12} {fps_str:<10} {params_str:<12}"
        print(row)
    
    print("="*100)


def main():
    """主函数"""
    args = parse_args()
    
    print("\n" + "="*80)
    print("消融实验结果汇总")
    print("="*80)
    print(f"结果目录: {args.results_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"基线实验: {args.baseline}")
    
    # 1. 扫描结果目录
    print("\n" + "="*80)
    print("步骤 1: 扫描结果目录")
    print("="*80)
    data = scan_results(args.results_dir)
    
    if not data:
        print("\n❌ 未找到任何实验结果")
        print("请先运行消融实验:")
        print("  python scripts/run_ablation.py")
        return 1
    
    print(f"\n✓ 扫描完成，找到 {len(data)} 个实验")
    for exp_name, seeds in data.items():
        print(f"  - {exp_name}: {len(seeds)} 个种子")
    
    # 2. 聚合指标
    print("\n" + "="*80)
    print("步骤 2: 聚合指标 (计算 mean ± std)")
    print("="*80)
    aggregated = aggregate_metrics(data)
    
    # 3. 计算相对提升
    print("\n" + "="*80)
    print("步骤 3: 计算相对 baseline 的提升")
    print("="*80)
    aggregated = calculate_relative_improvements(aggregated, args.baseline)
    
    # 4. 保存 CSV
    print("\n" + "="*80)
    print("步骤 4: 保存汇总结果")
    print("="*80)
    output_csv = Path(args.output_dir) / "ablation_summary.csv"
    save_summary_csv(aggregated, output_csv)
    
    # 5. 打印简要表格
    print_summary_table(aggregated)
    
    # 6. 保存完整数据（用于报告生成）
    output_json = Path(args.output_dir) / "ablation_summary.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)
    print(f"✓ 完整数据已保存: {output_json}")
    
    print("\n" + "="*80)
    print("✅ 汇总完成！")
    print("="*80)
    print("\n下一步：生成报告")
    print("  python scripts/make_report.py")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
