"""
对比多个模型的基准测试结果

用法:
    python scripts/compare_benchmarks.py
    python scripts/compare_benchmarks.py --output results/bench/comparison.csv
"""

import sys
from pathlib import Path
import json
import argparse
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def load_benchmark_results(bench_dir="results/bench"):
    """加载所有基准测试结果"""
    bench_path = Path(bench_dir)
    
    if not bench_path.exists():
        print(f"❌ 基准测试目录不存在: {bench_path}")
        return []
    
    results = []
    
    for json_file in bench_path.glob("*.json"):
        if json_file.name == "benchmark_list.txt":
            continue
        
        try:
            with open(json_file, encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取关键信息
            model_name = data["metadata"]["model_name"]
            params_M = data["model_complexity"]["params_M"]
            gflops = data["model_complexity"]["gflops"]
            latency_mean = data["latency"]["mean_ms"]
            latency_p50 = data["latency"]["p50_ms"]
            latency_p95 = data["latency"]["p95_ms"]
            fps_mean = data["throughput"]["fps_mean"]
            gpu_mem = data["memory"]["peak_gpu_mem_MB"]
            
            results.append({
                "Model": model_name,
                "Params (M)": round(params_M, 2),
                "GFLOPs": round(gflops, 2),
                "Latency Mean (ms)": round(latency_mean, 2),
                "Latency P50 (ms)": round(latency_p50, 2),
                "Latency P95 (ms)": round(latency_p95, 2),
                "FPS (mean)": round(fps_mean, 1),
                "GPU Mem (MB)": round(gpu_mem, 1),
            })
            
        except Exception as e:
            print(f"⚠️  加载 {json_file.name} 失败: {e}")
            continue
    
    return results


def compute_relative_metrics(df, baseline="baseline"):
    """计算相对于baseline的指标"""
    if baseline not in df["Model"].values:
        print(f"⚠️  未找到baseline模型: {baseline}")
        return df
    
    baseline_row = df[df["Model"] == baseline].iloc[0]
    
    # 计算相对值
    df["Params Δ (%)"] = ((df["Params (M)"] - baseline_row["Params (M)"]) / baseline_row["Params (M)"] * 100).round(1)
    df["Latency Δ (%)"] = ((df["Latency Mean (ms)"] - baseline_row["Latency Mean (ms)"]) / baseline_row["Latency Mean (ms)"] * 100).round(1)
    df["FPS Δ (%)"] = ((df["FPS (mean)"] - baseline_row["FPS (mean)"]) / baseline_row["FPS (mean)"] * 100).round(1)
    df["Mem Δ (%)"] = ((df["GPU Mem (MB)"] - baseline_row["GPU Mem (MB)"]) / baseline_row["GPU Mem (MB)"] * 100).round(1)
    
    return df


def print_comparison_table(df):
    """打印对比表格"""
    print("\n" + "="*120)
    print("模型性能基准测试对比")
    print("="*120)
    
    # 基础指标
    print("\n基础指标:")
    print("-"*120)
    basic_cols = ["Model", "Params (M)", "GFLOPs", "Latency Mean (ms)", "Latency P95 (ms)", "FPS (mean)", "GPU Mem (MB)"]
    print(df[basic_cols].to_string(index=False))
    
    # 相对指标（如果有）
    if "Params Δ (%)" in df.columns:
        print("\n相对于baseline的变化:")
        print("-"*120)
        delta_cols = ["Model", "Params Δ (%)", "Latency Δ (%)", "FPS Δ (%)", "Mem Δ (%)"]
        print(df[delta_cols].to_string(index=False))
    
    print("="*120)


def generate_summary(df):
    """生成摘要统计"""
    print("\n" + "="*80)
    print("摘要统计")
    print("="*80)
    
    print(f"\n模型数量: {len(df)}")
    
    print(f"\n参数量范围:")
    print(f"  - 最小: {df['Params (M)'].min():.2f}M ({df.loc[df['Params (M)'].idxmin(), 'Model']})")
    print(f"  - 最大: {df['Params (M)'].max():.2f}M ({df.loc[df['Params (M)'].idxmax(), 'Model']})")
    print(f"  - 平均: {df['Params (M)'].mean():.2f}M")
    
    print(f"\n延迟范围:")
    print(f"  - 最快: {df['Latency Mean (ms)'].min():.2f}ms ({df.loc[df['Latency Mean (ms)'].idxmin(), 'Model']})")
    print(f"  - 最慢: {df['Latency Mean (ms)'].max():.2f}ms ({df.loc[df['Latency Mean (ms)'].idxmax(), 'Model']})")
    print(f"  - 平均: {df['Latency Mean (ms)'].mean():.2f}ms")
    
    print(f"\nFPS范围:")
    print(f"  - 最高: {df['FPS (mean)'].max():.1f} ({df.loc[df['FPS (mean)'].idxmax(), 'Model']})")
    print(f"  - 最低: {df['FPS (mean)'].min():.1f} ({df.loc[df['FPS (mean)'].idxmin(), 'Model']})")
    print(f"  - 平均: {df['FPS (mean)'].mean():.1f}")
    
    print(f"\nGPU显存范围:")
    print(f"  - 最小: {df['GPU Mem (MB)'].min():.1f}MB ({df.loc[df['GPU Mem (MB)'].idxmin(), 'Model']})")
    print(f"  - 最大: {df['GPU Mem (MB)'].max():.1f}MB ({df.loc[df['GPU Mem (MB)'].idxmax(), 'Model']})")
    print(f"  - 平均: {df['GPU Mem (MB)'].mean():.1f}MB")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="对比模型基准测试结果")
    parser.add_argument("--bench_dir", default="results/bench", help="基准测试结果目录")
    parser.add_argument("--baseline", default="baseline", help="baseline模型名称")
    parser.add_argument("--output", help="输出CSV文件路径")
    parser.add_argument("--sort", default="params", choices=["params", "latency", "fps", "mem"],
                       help="排序依据")
    
    args = parser.parse_args()
    
    # 加载结果
    results = load_benchmark_results(args.bench_dir)
    
    if not results:
        print("❌ 未找到任何基准测试结果")
        print(f"\n请先运行基准测试:")
        print(f"  python scripts/benchmark_model.py --weights <model.pt>")
        return 1
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 排序
    sort_map = {
        "params": "Params (M)",
        "latency": "Latency Mean (ms)",
        "fps": "FPS (mean)",
        "mem": "GPU Mem (MB)"
    }
    df = df.sort_values(sort_map[args.sort])
    
    # 计算相对指标
    df = compute_relative_metrics(df, baseline=args.baseline)
    
    # 打印对比表
    print_comparison_table(df)
    
    # 生成摘要
    generate_summary(df)
    
    # 保存CSV
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✅ 对比结果已保存: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
