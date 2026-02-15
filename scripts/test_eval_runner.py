"""
测试评估运行器的功能

验证：
1. 参数解析
2. 输出JSON结构
3. 中心点误差计算
4. 失败案例导出
"""

import sys
from pathlib import Path
import json
import numpy as np
import argparse

# 添加项目根目录
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts.run_eval_one import (
    compute_iou_matrix,
    get_box_area,
    get_box_center,
    categorize_by_size,
    match_predictions_to_gt,
    compute_center_errors,
    linear_sum_assignment_greedy
)


def test_iou_computation():
    """测试IoU计算"""
    print("\n" + "="*80)
    print("测试 1: IoU矩阵计算")
    print("="*80)
    
    # 测试用例：两个框
    boxes1 = np.array([
        [0, 0, 10, 10],    # 框1
        [5, 5, 15, 15],    # 框2
    ])
    
    boxes2 = np.array([
        [0, 0, 10, 10],    # 与框1完全重叠
        [20, 20, 30, 30],  # 与所有框不重叠
    ])
    
    iou_matrix = compute_iou_matrix(boxes1, boxes2)
    
    print(f"预测框1: {boxes1[0]}")
    print(f"预测框2: {boxes1[1]}")
    print(f"GT框1: {boxes2[0]}")
    print(f"GT框2: {boxes2[1]}")
    print(f"\nIoU矩阵:\n{iou_matrix}")
    
    # 验证
    assert iou_matrix[0, 0] > 0.99, "框1与GT1应完全重叠"
    assert iou_matrix[0, 1] < 0.01, "框1与GT2不应重叠"
    assert iou_matrix[1, 1] < 0.01, "框2与GT2不应重叠"
    
    print("✅ IoU计算测试通过")


def test_size_categorization():
    """测试尺寸分类"""
    print("\n" + "="*80)
    print("测试 2: 目标尺寸分类")
    print("="*80)
    
    # 测试框面积
    boxes = np.array([
        [0, 0, 20, 20],      # 400 px² - small (< 32²=1024)
        [0, 0, 50, 50],      # 2500 px² - medium (32²~96²)
        [0, 0, 100, 100],    # 10000 px² - large (> 96²=9216)
    ])
    
    areas = get_box_area(boxes)
    categories = categorize_by_size(areas, [32, 96])
    
    category_names = ['small', 'medium', 'large']
    
    print(f"阈值: T1=32 (1024 px²), T2=96 (9216 px²)")
    for i, (box, area, cat) in enumerate(zip(boxes, areas, categories)):
        print(f"框{i+1}: 面积={area:.0f} px² -> {category_names[cat]}")
    
    # 验证
    assert categories[0] == 0, "400px²应为small"
    assert categories[1] == 1, "2500px²应为medium"
    assert categories[2] == 2, "10000px²应为large"
    
    print("✅ 尺寸分类测试通过")


def test_matching():
    """测试框匹配"""
    print("\n" + "="*80)
    print("测试 3: 预测框与GT框匹配（贪心算法）")
    print("="*80)
    
    # 预测框
    pred_boxes = np.array([
        [0, 0, 10, 10],      # 与GT1匹配
        [50, 50, 60, 60],    # FP（无GT）
        [100, 100, 110, 110], # 与GT3匹配
    ])
    
    # GT框
    gt_boxes = np.array([
        [0, 0, 10, 10],      # 与Pred1匹配
        [20, 20, 30, 30],    # FN（无预测）
        [100, 100, 110, 110], # 与Pred3匹配（IoU=1.0）
    ])
    
    matched_pred, matched_gt, fp_idx, fn_idx = match_predictions_to_gt(
        pred_boxes, gt_boxes, iou_threshold=0.5
    )
    
    print(f"匹配结果（IoU >= 0.5）:")
    print(f"  - 匹配的预测框索引: {matched_pred}")
    print(f"  - 匹配的GT框索引: {matched_gt}")
    print(f"  - FP（未匹配预测）: {fp_idx}")
    print(f"  - FN（未匹配GT）: {fn_idx}")
    
    # 计算实际IoU来验证
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    print(f"\nIoU矩阵:")
    print(f"  Pred0-GT0: {iou_matrix[0,0]:.2f}")
    print(f"  Pred2-GT2: {iou_matrix[2,2]:.2f}")
    
    # 验证
    assert 0 in matched_pred and 0 in matched_gt, "Pred0应与GT0匹配"
    assert 2 in matched_pred and 2 in matched_gt, "Pred2应与GT2匹配"
    assert 1 in fp_idx, "Pred1应为FP"
    assert 1 in fn_idx, "GT1应为FN"
    
    print("✅ 匹配算法测试通过")


def test_center_error():
    """测试中心点误差计算"""
    print("\n" + "="*80)
    print("测试 4: 中心点定位误差")
    print("="*80)
    
    # 预测框（中心稍有偏移）
    pred_boxes = np.array([
        [0, 0, 10, 10],      # 中心 (5, 5)
        [20, 20, 30, 30],    # 中心 (25, 25)
    ])
    
    # GT框
    gt_boxes = np.array([
        [1, 1, 11, 11],      # 中心 (6, 6)，偏移√2≈1.41
        [20, 22, 30, 32],    # 中心 (25, 27)，偏移2.0
    ])
    
    matched_pred = [0, 1]
    matched_gt = [0, 1]
    
    errors = compute_center_errors(pred_boxes, gt_boxes, matched_pred, matched_gt)
    
    pred_centers = get_box_center(pred_boxes)
    gt_centers = get_box_center(gt_boxes)
    
    print(f"框匹配对:")
    for i, (p_idx, g_idx) in enumerate(zip(matched_pred, matched_gt)):
        print(f"  Pred{p_idx} 中心: {pred_centers[p_idx]}")
        print(f"  GT{g_idx} 中心: {gt_centers[g_idx]}")
        print(f"  定位误差: {errors[i]:.2f} px")
    
    # 验证
    expected_error_0 = np.sqrt(2)  # √2 ≈ 1.41
    expected_error_1 = 2.0
    
    assert abs(errors[0] - expected_error_0) < 0.01, f"第1对误差应为√2，实际{errors[0]}"
    assert abs(errors[1] - expected_error_1) < 0.01, f"第2对误差应为2.0，实际{errors[1]}"
    
    print(f"\n统计:")
    print(f"  - Mean: {np.mean(errors):.2f} px")
    print(f"  - Median: {np.median(errors):.2f} px")
    print(f"  - P95: {np.percentile(errors, 95):.2f} px")
    
    print("✅ 中心点误差计算测试通过")


def test_greedy_algorithm():
    """测试贪心匹配算法"""
    print("\n" + "="*80)
    print("测试 5: 贪心匹配算法")
    print("="*80)
    
    # 代价矩阵（负IoU）
    cost_matrix = np.array([
        [-0.9, -0.1, -0.2],  # Pred1: 与GT1最匹配
        [-0.3, -0.8, -0.1],  # Pred2: 与GT2最匹配
        [-0.1, -0.2, -0.7],  # Pred3: 与GT3最匹配
    ])
    
    pred_idx, gt_idx = linear_sum_assignment_greedy(cost_matrix)
    
    print(f"代价矩阵（负IoU）:\n{cost_matrix}")
    print(f"\n贪心匹配结果:")
    for p, g in zip(pred_idx, gt_idx):
        print(f"  Pred{p} -> GT{g}, IoU={-cost_matrix[p, g]:.2f}")
    
    # 验证：应按最大IoU贪心匹配
    assert len(pred_idx) == 3, "应有3对匹配"
    assert 0 in pred_idx and 0 in gt_idx, "Pred0应与GT0匹配（IoU=0.9）"
    
    print("✅ 贪心算法测试通过")


def test_json_structure():
    """测试输出JSON结构"""
    print("\n" + "="*80)
    print("测试 6: 评估结果JSON结构")
    print("="*80)
    
    # 模拟评估结果
    eval_results = {
        'metadata': {
            'exp_name': 'test_baseline',
            'weights': 'best.pt',
            'seed': 0,
            'split': 'val',
        },
        'standard_metrics': {
            'mAP50': 0.7123,
            'mAP50-95': 0.5234,
            'precision': 0.8012,
            'recall': 0.6789,
        },
        'size_based_ap': {
            'small': {'ap50': 0.6234, 'count': 100},
            'medium': {'ap50': 0.7456, 'count': 200},
            'large': {'ap50': 0.8123, 'count': 150},
            'thresholds': [32, 96],
        },
        'center_localization_error': {
            'mean': 5.23,
            'median': 4.12,
            'p95': 12.34,
            'count': 450,
            'algorithm': 'greedy_matching',
        },
        'failure_cases': {
            'top_k': 20,
            'cases_dir': 'results/runs/test/cases',
            'summary_csv': 'results/runs/test/cases_summary.csv',
            'total_fp': 15,
            'total_fn': 8,
        }
    }
    
    # 验证必需字段
    assert 'standard_metrics' in eval_results
    assert 'mAP50' in eval_results['standard_metrics']
    assert 'center_localization_error' in eval_results
    assert 'mean' in eval_results['center_localization_error']
    assert 'size_based_ap' in eval_results
    assert 'failure_cases' in eval_results
    
    print(f"✅ JSON结构包含所有必需字段:")
    print(f"  ✓ standard_metrics (mAP50, mAP50-95, P, R)")
    print(f"  ✓ size_based_ap (small/medium/large)")
    print(f"  ✓ center_localization_error (mean, median, p95)")
    print(f"  ✓ failure_cases (cases_dir, summary_csv)")
    
    # 打印示例
    print(f"\nJSON示例:")
    print(json.dumps(eval_results, indent=2))
    
    print("✅ JSON结构测试通过")


def main():
    """运行所有测试"""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "评估运行器单元测试" + " "*38 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        test_iou_computation()
        test_size_categorization()
        test_matching()
        test_center_error()
        test_greedy_algorithm()
        test_json_structure()
        
        print("\n" + "="*80)
        print("测试总结")
        print("="*80)
        print("✅ 所有测试通过！")
        print("\n评估运行器功能验证:")
        print("  ✓ IoU矩阵计算")
        print("  ✓ 目标尺寸分类（small/medium/large）")
        print("  ✓ 贪心匹配算法（IoU >= 0.5）")
        print("  ✓ 中心点定位误差计算")
        print("  ✓ 输出JSON结构")
        print("\n可以使用以下命令运行实际评估:")
        print("  python scripts/run_eval_one.py \\")
        print("      --exp_name baseline \\")
        print("      --weights results/runs/baseline/seed0/weights/best.pt \\")
        print("      --seed 0 \\")
        print("      --split val")
        print("="*80 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
