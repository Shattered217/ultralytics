"""
测试结果汇总和报告生成功能

创建模拟数据并测试：
1. aggregate_results.py 数据汇总
2. make_report.py 报告生成
"""

import sys
from pathlib import Path
import json
import shutil

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def create_mock_data():
    """创建模拟的实验结果数据"""
    print("\n" + "="*80)
    print("创建模拟实验数据")
    print("="*80)
    
    # 定义模拟实验
    experiments = {
        'baseline': {'params': 3162272, 'gflops': 8.2},
        'ghost': {'params': 2140000, 'gflops': 5.8},
        'eca': {'params': 3165000, 'gflops': 8.3},
        'p2': {'params': 3350000, 'gflops': 9.1},
    }
    
    seeds = [0, 1, 2]
    
    # 为每个实验创建模拟数据
    for exp_name, exp_info in experiments.items():
        print(f"\n创建实验: {exp_name}")
        
        for seed in seeds:
            print(f"  种子: {seed}")
            
            # 创建目录结构
            seed_name = f"seed{seed}"
            
            # 1. 训练输出目录
            train_dir = Path("results/runs") / exp_name / seed_name / "weights"
            train_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建空权重文件（用于检测）
            (train_dir / "best.pt").touch()
            
            # 2. 评估输出目录 (test split)
            eval_dir = Path("results/evals") / f"{exp_name}_{seed_name}_test"
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成模拟评估指标
            # 添加一些随机性
            import random
            random.seed(seed + hash(exp_name))
            
            # 基线性能
            base_map50 = 0.750
            base_map5095 = 0.540
            base_ap_small = 0.320
            base_ap_medium = 0.650
            base_ap_large = 0.810
            base_center_err = 3.5
            
            # 不同实验的性能变化
            if exp_name == 'baseline':
                map50 = base_map50 + random.uniform(-0.01, 0.01)
                map5095 = base_map5095 + random.uniform(-0.01, 0.01)
                ap_small = base_ap_small + random.uniform(-0.01, 0.01)
                ap_medium = base_ap_medium + random.uniform(-0.01, 0.01)
                ap_large = base_ap_large + random.uniform(-0.01, 0.01)
                center_err = base_center_err + random.uniform(-0.2, 0.2)
            elif exp_name == 'ghost':
                # Ghost: 速度快，精度略降
                map50 = base_map50 - 0.02 + random.uniform(-0.01, 0.01)
                map5095 = base_map5095 - 0.015 + random.uniform(-0.01, 0.01)
                ap_small = base_ap_small - 0.01 + random.uniform(-0.01, 0.01)
                ap_medium = base_ap_medium - 0.02 + random.uniform(-0.01, 0.01)
                ap_large = base_ap_large - 0.01 + random.uniform(-0.01, 0.01)
                center_err = base_center_err + 0.3 + random.uniform(-0.2, 0.2)
            elif exp_name == 'eca':
                # ECA: 精度提升，速度略降
                map50 = base_map50 + 0.015 + random.uniform(-0.01, 0.01)
                map5095 = base_map5095 + 0.012 + random.uniform(-0.01, 0.01)
                ap_small = base_ap_small + 0.01 + random.uniform(-0.01, 0.01)
                ap_medium = base_ap_medium + 0.015 + random.uniform(-0.01, 0.01)
                ap_large = base_ap_large + 0.012 + random.uniform(-0.01, 0.01)
                center_err = base_center_err - 0.4 + random.uniform(-0.2, 0.2)
            elif exp_name == 'p2':
                # P2: 小目标提升，速度降低
                map50 = base_map50 + 0.010 + random.uniform(-0.01, 0.01)
                map5095 = base_map5095 + 0.008 + random.uniform(-0.01, 0.01)
                ap_small = base_ap_small + 0.05 + random.uniform(-0.01, 0.01)
                ap_medium = base_ap_medium + 0.01 + random.uniform(-0.01, 0.01)
                ap_large = base_ap_large + 0.005 + random.uniform(-0.01, 0.01)
                center_err = base_center_err - 0.2 + random.uniform(-0.2, 0.2)
            
            # 保存标准指标
            metrics = {
                'mAP50': float(map50),
                'mAP50-95': float(map5095),
                'Precision': float(0.82 + random.uniform(-0.02, 0.02)),
                'Recall': float(0.75 + random.uniform(-0.02, 0.02)),
            }
            with open(eval_dir / "metrics.json", 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            
            # 保存尺度指标
            size_metrics = {
                'AP_small': float(ap_small),
                'AP_medium': float(ap_medium),
                'AP_large': float(ap_large),
            }
            with open(eval_dir / "size_metrics.json", 'w', encoding='utf-8') as f:
                json.dump(size_metrics, f, indent=2)
            
            # 保存中心点误差
            center_errors = {
                'mean_error_pixels': float(center_err),
                'median_error_pixels': float(center_err * 0.7),
                'max_error_pixels': float(center_err * 4.5),
            }
            with open(eval_dir / "center_errors.json", 'w', encoding='utf-8') as f:
                json.dump(center_errors, f, indent=2)
            
            # 3. 基准测试输出
            bench_dir = Path("results/benchmarks")
            bench_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成模拟基准测试指标
            base_fps = 82.0
            
            if exp_name == 'ghost':
                fps = base_fps * 1.25 + random.uniform(-2, 2)
            elif exp_name == 'eca':
                fps = base_fps * 0.92 + random.uniform(-2, 2)
            elif exp_name == 'p2':
                fps = base_fps * 0.85 + random.uniform(-2, 2)
            else:
                fps = base_fps + random.uniform(-2, 2)
            
            latency = 1000.0 / fps
            
            bench_metrics = {
                'model_params': exp_info['params'],
                'model_gflops': exp_info['gflops'],
                'latency_mean_ms': float(latency),
                'latency_std_ms': float(latency * 0.05),
                'latency_p50_ms': float(latency * 0.98),
                'latency_p95_ms': float(latency * 1.15),
                'latency_p99_ms': float(latency * 1.25),
                'fps': float(fps),
                'memory_allocated_mb': float(200 + random.uniform(-20, 20)),
                'memory_peak_mb': float(450 + random.uniform(-30, 30)),
            }
            
            bench_file = bench_dir / f"{exp_name}_{seed_name}_benchmark.json"
            with open(bench_file, 'w', encoding='utf-8') as f:
                json.dump(bench_metrics, f, indent=2)
    
    print("\n✓ 模拟数据创建完成")


def test_aggregate_results():
    """测试结果汇总"""
    print("\n" + "="*80)
    print("测试 aggregate_results.py")
    print("="*80)
    
    import subprocess
    
    cmd = [
        sys.executable,
        "scripts/aggregate_results.py",
        "--results_dir", "results",
        "--output_dir", "results/summary",
    ]
    
    print(f"执行命令: {' '.join(cmd)}\n")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    print(result.stdout)
    
    if result.returncode == 0:
        print("✅ aggregate_results.py 执行成功")
        
        # 检查输出文件
        csv_file = Path("results/summary/ablation_summary.csv")
        json_file = Path("results/summary/ablation_summary.json")
        
        if csv_file.exists():
            print(f"✓ CSV 文件已生成: {csv_file}")
            print(f"  大小: {csv_file.stat().st_size} bytes")
            
            # 读取并显示前几行
            with open(csv_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:5]
            print(f"  前 {len(lines)} 行:")
            for line in lines:
                print(f"    {line.rstrip()}")
        else:
            print(f"✗ CSV 文件未生成: {csv_file}")
            return False
        
        if json_file.exists():
            print(f"✓ JSON 文件已生成: {json_file}")
        else:
            print(f"✗ JSON 文件未生成: {json_file}")
            return False
        
        return True
    else:
        print(f"❌ aggregate_results.py 执行失败")
        print(f"stderr: {result.stderr}")
        return False


def test_make_report():
    """测试报告生成"""
    print("\n" + "="*80)
    print("测试 make_report.py")
    print("="*80)
    
    import subprocess
    
    cmd = [
        sys.executable,
        "scripts/make_report.py",
        "--summary", "results/summary/ablation_summary.json",
        "--output_dir", "results/summary",
    ]
    
    print(f"执行命令: {' '.join(cmd)}\n")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    print(result.stdout)
    
    if result.returncode == 0:
        print("✅ make_report.py 执行成功")
        
        # 检查输出文件
        md_file = Path("results/summary/ablation_report.md")
        plots = [
            Path("results/summary/plot_map_vs_fps.png"),
            Path("results/summary/plot_ap_small_vs_fps.png"),
            Path("results/summary/plot_center_err_vs_fps.png"),
        ]
        
        if md_file.exists():
            print(f"✓ Markdown 报告已生成: {md_file}")
            print(f"  大小: {md_file.stat().st_size} bytes")
            
            # 统计行数
            with open(md_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"  行数: {len(lines)}")
        else:
            print(f"✗ Markdown 报告未生成: {md_file}")
            return False
        
        # 检查图表
        all_plots_exist = True
        for plot_file in plots:
            if plot_file.exists():
                print(f"✓ 图表已生成: {plot_file.name}")
                print(f"  大小: {plot_file.stat().st_size} bytes")
            else:
                print(f"✗ 图表未生成: {plot_file}")
                all_plots_exist = False
        
        return all_plots_exist
    else:
        print(f"❌ make_report.py 执行失败")
        print(f"stderr: {result.stderr}")
        return False


def cleanup_mock_data():
    """清理模拟数据"""
    print("\n" + "="*80)
    print("清理测试数据")
    print("="*80)
    
    # 可选：保留数据供查看
    response = input("是否删除模拟数据？(y/N): ").strip().lower()
    
    if response == 'y':
        paths_to_remove = [
            Path("results/runs"),
            Path("results/evals"),
            Path("results/benchmarks"),
            Path("results/summary"),
        ]
        
        for path in paths_to_remove:
            if path.exists():
                shutil.rmtree(path)
                print(f"✓ 已删除: {path}")
    else:
        print("保留模拟数据，可手动查看:")
        print("  - results/summary/ablation_summary.csv")
        print("  - results/summary/ablation_report.md")
        print("  - results/summary/plot_*.png")


def main():
    """主函数"""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "结果汇总与报告生成测试" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        ("创建模拟数据", create_mock_data),
        ("测试 aggregate_results.py", test_aggregate_results),
        ("测试 make_report.py", test_make_report),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if test_func == create_mock_data:
                # 创建数据不返回 bool
                test_func()
                results.append((test_name, True))
            else:
                success = test_func()
                results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n通过率: {passed}/{total} ({100*passed/total:.0f}%)")
    
    if passed == total:
        print("\n✅ 所有测试通过！")
        print("\n生成的文件:")
        print("  📄 results/summary/ablation_summary.csv  - 汇总表格")
        print("  📄 results/summary/ablation_summary.json - 完整数据")
        print("  📄 results/summary/ablation_report.md    - Markdown 报告")
        print("  📊 results/summary/plot_map_vs_fps.png   - mAP vs FPS")
        print("  📊 results/summary/plot_ap_small_vs_fps.png - AP_small vs FPS")
        print("  📊 results/summary/plot_center_err_vs_fps.png - 中心误差 vs FPS")
        print("\n查看报告:")
        print("  cat results/summary/ablation_report.md")
        print("="*80 + "\n")
        
        # 清理
        cleanup_mock_data()
        
        return 0
    else:
        print(f"\n❌ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
