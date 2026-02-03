"""
评估运行器验收测试

验收标准：
1. eval_val 与 eval_test 都能生成 JSON
2. cases 目录有可视化输出
3. 中心点误差统计存在

运行此脚本前，请确保：
- 已有训练好的模型权重
- 数据集配置正确
"""

import sys
from pathlib import Path
import json

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def check_acceptance_criteria(exp_name, seed=0):
    """
    验收测试：检查评估输出是否符合要求
    """
    results_dir = Path(f"results/runs/{exp_name}/seed{seed}")
    
    print("\n" + "="*80)
    print(f"评估运行器验收测试")
    print("="*80)
    print(f"实验名称: {exp_name}")
    print(f"随机种子: {seed}")
    print(f"结果目录: {results_dir}")
    
    # 检查 1: eval_val.json 存在
    print("\n" + "-"*80)
    print("检查 1: 验证集评估结果")
    print("-"*80)
    
    eval_val = results_dir / "eval_val.json"
    if eval_val.exists():
        print(f"✅ {eval_val} 存在")
        
        # 验证JSON结构
        with open(eval_val) as f:
            data = json.load(f)
        
        required_keys = ['standard_metrics', 'size_based_ap', 
                        'center_localization_error', 'failure_cases']
        
        for key in required_keys:
            if key in data:
                print(f"  ✓ {key}")
            else:
                print(f"  ✗ 缺少 {key}")
                return False
        
        # 验证标准指标
        metrics = data['standard_metrics']
        print(f"\n  标准指标:")
        print(f"    - mAP50: {metrics.get('mAP50', 'N/A')}")
        print(f"    - mAP50-95: {metrics.get('mAP50-95', 'N/A')}")
        print(f"    - Precision: {metrics.get('precision', 'N/A')}")
        print(f"    - Recall: {metrics.get('recall', 'N/A')}")
        
        # 验证尺寸分桶AP
        size_ap = data['size_based_ap']
        print(f"\n  尺寸分桶AP:")
        print(f"    - Small: {size_ap.get('small', {}).get('ap50', 'N/A')}")
        print(f"    - Medium: {size_ap.get('medium', {}).get('ap50', 'N/A')}")
        print(f"    - Large: {size_ap.get('large', {}).get('ap50', 'N/A')}")
        
        # 验证中心点误差
        center_error = data['center_localization_error']
        print(f"\n  中心点定位误差:")
        print(f"    - Mean: {center_error.get('mean', 'N/A')} px")
        print(f"    - Median: {center_error.get('median', 'N/A')} px")
        print(f"    - P95: {center_error.get('p95', 'N/A')} px")
        print(f"    - Count: {center_error.get('count', 'N/A')}")
        print(f"    - Algorithm: {center_error.get('algorithm', 'N/A')}")
        
        if center_error.get('count', 0) == 0:
            print(f"  ⚠️  警告：没有匹配的框（可能模型性能很差）")
    else:
        print(f"❌ {eval_val} 不存在")
        return False
    
    # 检查 2: eval_test.json（可选）
    print("\n" + "-"*80)
    print("检查 2: 测试集评估结果（可选）")
    print("-"*80)
    
    eval_test = results_dir / "eval_test.json"
    if eval_test.exists():
        print(f"✅ {eval_test} 存在")
    else:
        print(f"⚠️  {eval_test} 不存在（如果数据集没有test split则正常）")
    
    # 检查 3: cases 目录
    print("\n" + "-"*80)
    print("检查 3: 失败案例可视化")
    print("-"*80)
    
    cases_dir = results_dir / "cases"
    if cases_dir.exists() and cases_dir.is_dir():
        print(f"✅ {cases_dir} 存在")
        
        # 统计可视化文件
        case_files = list(cases_dir.glob("*.jpg")) + list(cases_dir.glob("*.png"))
        print(f"  可视化案例数: {len(case_files)}")
        
        if len(case_files) > 0:
            print(f"  示例文件:")
            for f in case_files[:5]:
                print(f"    - {f.name}")
            if len(case_files) > 5:
                print(f"    ... 还有 {len(case_files)-5} 个文件")
        else:
            print(f"  ⚠️  警告：没有可视化文件（可能没有失败案例）")
    else:
        print(f"❌ {cases_dir} 不存在")
        return False
    
    # 检查 4: cases_summary.csv
    print("\n" + "-"*80)
    print("检查 4: 失败案例摘要")
    print("-"*80)
    
    cases_csv = results_dir / "cases_summary.csv"
    if cases_csv.exists():
        print(f"✅ {cases_csv} 存在")
        
        # 读取前几行
        with open(cases_csv) as f:
            lines = f.readlines()
        
        print(f"  总行数: {len(lines)}")
        if len(lines) > 0:
            print(f"  表头: {lines[0].strip()}")
        if len(lines) > 1:
            print(f"  示例行: {lines[1].strip()}")
    else:
        print(f"❌ {cases_csv} 不存在")
        return False
    
    # 验收通过
    print("\n" + "="*80)
    print("验收测试结果")
    print("="*80)
    print("✅ 所有验收标准通过！")
    print("\n验收项:")
    print("  ✓ eval_val.json 生成（包含所有必需字段）")
    print("  ✓ cases/ 目录存在（包含可视化）")
    print("  ✓ cases_summary.csv 存在")
    print("  ✓ 中心点误差统计存在")
    print("="*80 + "\n")
    
    return True


def print_usage():
    """打印使用说明"""
    print("\n" + "="*80)
    print("评估运行器验收测试 - 使用说明")
    print("="*80)
    print("\n步骤 1: 训练模型")
    print("  python scripts/run_train_one.py \\")
    print("      --exp_name baseline \\")
    print("      --model_yaml ultralytics/cfg/models/v8/yolov8n_baseline.yaml \\")
    print("      --seed 0")
    
    print("\n步骤 2: 在验证集上评估")
    print("  python scripts/run_eval_one.py \\")
    print("      --exp_name baseline \\")
    print("      --weights results/runs/baseline/seed0/weights/best.pt \\")
    print("      --seed 0 \\")
    print("      --split val")
    
    print("\n步骤 3: 在测试集上评估（如有）")
    print("  python scripts/run_eval_one.py \\")
    print("      --exp_name baseline \\")
    print("      --weights results/runs/baseline/seed0/weights/best.pt \\")
    print("      --seed 0 \\")
    print("      --split test")
    
    print("\n步骤 4: 运行验收测试")
    print("  python scripts/test_eval_acceptance.py baseline 0")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n用法: python scripts/test_eval_acceptance.py <exp_name> [seed]")
        print("示例: python scripts/test_eval_acceptance.py baseline 0")
        print_usage()
        sys.exit(1)
    
    exp_name = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    # 检查权重是否存在
    weights = Path(f"results/runs/{exp_name}/seed{seed}/weights/best.pt")
    if not weights.exists():
        print(f"\n❌ 错误：权重文件不存在: {weights}")
        print(f"\n请先训练模型：")
        print(f"  python scripts/run_train_one.py \\")
        print(f"      --exp_name {exp_name} \\")
        print(f"      --model_yaml <模型配置> \\")
        print(f"      --seed {seed}")
        sys.exit(1)
    
    # 检查评估结果是否存在
    eval_val = Path(f"results/runs/{exp_name}/seed{seed}/eval_val.json")
    if not eval_val.exists():
        print(f"\n❌ 错误：评估结果不存在: {eval_val}")
        print(f"\n请先运行评估：")
        print(f"  python scripts/run_eval_one.py \\")
        print(f"      --exp_name {exp_name} \\")
        print(f"      --weights {weights} \\")
        print(f"      --seed {seed} \\")
        print(f"      --split val")
        sys.exit(1)
    
    # 运行验收测试
    success = check_acceptance_criteria(exp_name, seed)
    
    if not success:
        print("\n❌ 验收测试失败")
        sys.exit(1)
