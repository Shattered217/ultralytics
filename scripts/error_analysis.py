"""
深度错误分析脚本

用于生成科研论文级别的不足分析材料，包括：
1. 混淆矩阵（按类别）
2. 每类AP统计
3. 误检模式分析（Top-20错误样例）
4. 遮挡/堆叠代理变量分析

使用方法：
    python scripts/error_analysis.py --weights results/runs/baseline/seed0/weights/best.pt --split test
    python scripts/error_analysis.py --weights results/runs/ghost/seed0/weights/best.pt --split test --conf 0.25
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import defaultdict, Counter
import cv2
from tqdm import tqdm
import csv
import os

# Windows UTF-8 输出修复
if sys.platform == "win32":
    import codecs
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="深度错误分析")
    parser.add_argument("--weights", type=str, required=True, help="模型权重路径")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="数据集划分")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou_match", type=float, default=0.5, help="IoU匹配阈值")
    parser.add_argument("--device", type=str, default="0", help="设备")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（默认自动推断）")
    parser.add_argument("--save_cases", action="store_true", help="保存错误案例图像")
    parser.add_argument("--top_k", type=int, default=20, help="保存Top-K错误样例")
    return parser.parse_args()


def load_model(weights, device='0'):
    """加载YOLO模型"""
    from ultralytics import YOLO
    
    print(f"加载模型: {weights}")
    model = YOLO(weights)
    model.to(f'cuda:{device}' if device.isdigit() else device)
    
    return model


def compute_iou(box1, box2):
    """计算IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    return iou


def compute_bbox_overlap(box1, box2):
    """计算两个bbox的重叠率（相对于box1的面积）"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    
    overlap_ratio = inter_area / (box1_area + 1e-6)
    return overlap_ratio


def match_predictions_to_ground_truth(predictions, ground_truths, iou_threshold=0.5):
    """
    匹配预测框和真实框
    
    返回：
    - matches: list of (pred_idx, gt_idx, iou)
    - fp_indices: 假阳性预测索引
    - fn_indices: 假阴性真实框索引
    """
    if len(predictions) == 0:
        return [], [], list(range(len(ground_truths)))
    
    if len(ground_truths) == 0:
        return [], list(range(len(predictions))), []
    
    # 计算IoU矩阵
    iou_matrix = np.zeros((len(predictions), len(ground_truths)))
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truths):
            iou_matrix[i, j] = compute_iou(pred['bbox'], gt['bbox'])
    
    # 贪心匹配（按置信度排序）
    matches = []
    matched_gt = set()
    matched_pred = set()
    
    # 按置信度降序排序预测
    pred_indices = sorted(range(len(predictions)), key=lambda i: predictions[i]['conf'], reverse=True)
    
    for pred_idx in pred_indices:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in range(len(ground_truths)):
            if gt_idx in matched_gt:
                continue
            
            iou = iou_matrix[pred_idx, gt_idx]
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_gt_idx >= 0 and predictions[pred_idx]['class'] == ground_truths[best_gt_idx]['class']:
            matches.append((pred_idx, best_gt_idx, best_iou))
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)
    
    # 未匹配的为FP和FN
    fp_indices = [i for i in range(len(predictions)) if i not in matched_pred]
    fn_indices = [i for i in range(len(ground_truths)) if i not in matched_gt]
    
    return matches, fp_indices, fn_indices


def run_inference_and_analyze(model, data_yaml, split, conf_threshold, iou_match_threshold):
    """
    运行推理并收集分析数据
    
    返回：
    - confusion_data: 混淆矩阵数据
    - per_class_results: 每类AP
    - fp_patterns: 误检模式
    - occlusion_data: 遮挡分析数据
    """
    import yaml
    from ultralytics.data.augment import LetterBox
    from PIL import Image
    
    # 读取数据配置
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # 获取类别名称
    class_names = data_config['names']
    num_classes = len(class_names)
    
    # 获取数据路径
    data_root = Path(data_config['path'])
    split_dir = data_root / 'images' / split
    label_dir = data_root / 'labels' / split
    
    if not split_dir.exists():
        print(f"❌ 数据目录不存在: {split_dir}")
        return None, None, None, None
    
    # 获取所有图像
    image_files = list(split_dir.glob('*.jpg')) + list(split_dir.glob('*.png'))
    print(f"找到 {len(image_files)} 张图像")
    
    # 初始化统计数据
    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)  # +1 for background
    per_class_tp = defaultdict(list)
    per_class_fp = defaultdict(list)
    per_class_fn = defaultdict(int)
    per_class_gt_count = defaultdict(int)
    
    fp_patterns = []  # (pred_class, gt_class, conf, image_path, bbox)
    occlusion_data = []  # (overlap_ratio, density, is_detected, class, image_path)
    
    # 推理每张图像
    print("\n运行推理...")
    for img_path in tqdm(image_files, desc="推理进度"):
        # 加载真实标签
        label_path = label_dir / (img_path.stem + '.txt')
        ground_truths = []
        
        if label_path.exists():
            img = Image.open(img_path)
            img_w, img_h = img.size
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:5])
                        
                        # YOLO格式转换为xyxy
                        x1 = (x_center - w / 2) * img_w
                        y1 = (y_center - h / 2) * img_h
                        x2 = (x_center + w / 2) * img_w
                        y2 = (y_center + h / 2) * img_h
                        
                        ground_truths.append({
                            'class': cls,
                            'bbox': [x1, y1, x2, y2],
                            'image_path': str(img_path),
                        })
                        
                        per_class_gt_count[cls] += 1
        
        # 运行推理
        results = model(str(img_path), conf=conf_threshold, verbose=False)
        result = results[0]
        
        # 提取预测
        predictions = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confs, classes):
                predictions.append({
                    'class': cls,
                    'conf': conf,
                    'bbox': box.tolist(),
                    'image_path': str(img_path),
                })
        
        # 匹配预测和真实框
        matches, fp_indices, fn_indices = match_predictions_to_ground_truth(
            predictions, ground_truths, iou_match_threshold
        )
        
        # 更新混淆矩阵和统计
        for pred_idx, gt_idx, iou in matches:
            pred_cls = predictions[pred_idx]['class']
            gt_cls = ground_truths[gt_idx]['class']
            
            confusion_matrix[gt_cls, pred_cls] += 1
            per_class_tp[pred_cls].append(predictions[pred_idx]['conf'])
        
        # 记录FP（假阳性）
        for pred_idx in fp_indices:
            pred = predictions[pred_idx]
            pred_cls = pred['class']
            
            # 找到最接近的GT类别
            best_iou = 0
            best_gt_cls = None
            for gt in ground_truths:
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_cls = gt['class']
            
            if best_gt_cls is not None:
                confusion_matrix[best_gt_cls, pred_cls] += 1
                fp_patterns.append({
                    'pred_class': pred_cls,
                    'gt_class': best_gt_cls,
                    'conf': pred['conf'],
                    'iou': best_iou,
                    'image_path': str(img_path),
                    'bbox': pred['bbox'],
                })
            else:
                # 背景误检
                confusion_matrix[num_classes, pred_cls] += 1
            
            per_class_fp[pred_cls].append(pred['conf'])
        
        # 记录FN（假阴性）
        for gt_idx in fn_indices:
            gt = ground_truths[gt_idx]
            gt_cls = gt['class']
            
            confusion_matrix[gt_cls, num_classes] += 1  # FN归入background列
            per_class_fn[gt_cls] += 1
        
        # 遮挡分析：计算每个GT与其他GT的重叠率
        if len(ground_truths) > 1:
            for i, gt in enumerate(ground_truths):
                max_overlap = 0
                for j, other_gt in enumerate(ground_truths):
                    if i != j:
                        overlap = compute_bbox_overlap(gt['bbox'], other_gt['bbox'])
                        max_overlap = max(max_overlap, overlap)
                
                # 计算目标密度（归一化）
                density = len(ground_truths) / (img_w * img_h / 1e6)  # per megapixel
                
                # 判断是否被检测
                is_detected = i not in fn_indices
                
                occlusion_data.append({
                    'overlap_ratio': max_overlap,
                    'density': density,
                    'is_detected': is_detected,
                    'class': gt['class'],
                    'image_path': str(img_path),
                })
    
    # 计算每类AP
    per_class_ap = {}
    for cls in range(num_classes):
        tp_confs = per_class_tp[cls]
        fp_confs = per_class_fp[cls]
        n_gt = per_class_gt_count[cls]
        
        if n_gt == 0:
            per_class_ap[cls] = 0.0
            continue
        
        # 合并TP和FP，按置信度排序
        all_detections = [(conf, True) for conf in tp_confs] + [(conf, False) for conf in fp_confs]
        all_detections.sort(key=lambda x: x[0], reverse=True)
        
        # 计算PR曲线
        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []
        
        for conf, is_tp in all_detections:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            recall = tp_cumsum / (n_gt + 1e-6)
            
            precisions.append(precision)
            recalls.append(recall)
        
        # 计算AP（11点插值）
        ap = 0
        for t in np.linspace(0, 1, 11):
            p_interp = max([p for p, r in zip(precisions, recalls) if r >= t] + [0])
            ap += p_interp / 11
        
        per_class_ap[cls] = ap
    
    print(f"\n✓ 推理完成")
    print(f"  - 总图像数: {len(image_files)}")
    print(f"  - 总GT数: {sum(per_class_gt_count.values())}")
    print(f"  - 总预测数: {sum(len(per_class_tp[c]) + len(per_class_fp[c]) for c in range(num_classes))}")
    print(f"  - 总TP数: {sum(len(per_class_tp[c]) for c in range(num_classes))}")
    print(f"  - 总FP数: {sum(len(per_class_fp[c]) for c in range(num_classes))}")
    print(f"  - 总FN数: {sum(per_class_fn[c] for c in range(num_classes))}")
    
    return confusion_matrix, per_class_ap, fp_patterns, occlusion_data, class_names


def plot_confusion_matrix(confusion_matrix, class_names, output_path):
    """绘制混淆矩阵"""
    # 添加背景类
    labels = class_names + ['Background']
    
    # 归一化（按行）
    cm_normalized = confusion_matrix.astype(float)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    cm_normalized = cm_normalized / row_sums
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={'label': 'Normalized Count'},
        square=True,
    )
    
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 混淆矩阵已保存: {output_path}")


def save_per_class_ap(per_class_ap, class_names, output_path):
    """保存每类AP"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Class Name', 'AP'])
        
        for cls, ap in sorted(per_class_ap.items()):
            class_name = class_names[cls] if cls < len(class_names) else 'Unknown'
            writer.writerow([cls, class_name, f'{ap:.4f}'])
        
        # 计算mAP
        mean_ap = np.mean(list(per_class_ap.values()))
        writer.writerow(['', 'mAP', f'{mean_ap:.4f}'])
    
    print(f"✓ 每类AP已保存: {output_path}")


def analyze_fp_patterns(fp_patterns, class_names, output_path, top_k=20):
    """
    分析误检模式并生成报告
    
    输出：
    1. 最常见误检对（pred_class -> gt_class）
    2. Top-K错误样例
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 统计误检对
    error_pairs = Counter()
    for fp in fp_patterns:
        pred_cls = fp['pred_class']
        gt_cls = fp['gt_class']
        pair_key = (pred_cls, gt_cls)
        error_pairs[pair_key] += 1
    
    # 按置信度排序，获取Top-K
    fp_patterns_sorted = sorted(fp_patterns, key=lambda x: x['conf'], reverse=True)
    top_k_fps = fp_patterns_sorted[:top_k]
    
    # 写入Markdown
    lines = []
    lines.append("# 误检模式分析（False Positive Patterns）")
    lines.append("")
    lines.append(f"生成时间: {Path(__file__).name}")
    lines.append("")
    
    # 1. 最常见误检对
    lines.append("## 1. 最常见误检对 (Top-20)")
    lines.append("")
    lines.append("误检对表示：模型预测为类别A，但实际为类别B")
    lines.append("")
    lines.append("| 排名 | 预测类别 → 真实类别 | 出现次数 | 百分比 |")
    lines.append("|------|---------------------|----------|--------|")
    
    total_fps = len(fp_patterns)
    for rank, ((pred_cls, gt_cls), count) in enumerate(error_pairs.most_common(20), 1):
        pred_name = class_names[pred_cls] if pred_cls < len(class_names) else 'Unknown'
        gt_name = class_names[gt_cls] if gt_cls < len(class_names) else 'Unknown'
        percentage = count / total_fps * 100
        lines.append(f"| {rank} | {pred_name} ({pred_cls}) → {gt_name} ({gt_cls}) | {count} | {percentage:.2f}% |")
    
    lines.append("")
    
    # 2. Top-K错误样例
    lines.append(f"## 2. Top-{top_k} 高置信度误检样例")
    lines.append("")
    lines.append("按置信度降序排列的误检样例（这些是模型最\"确信\"但错误的预测）")
    lines.append("")
    lines.append("| 排名 | 置信度 | 预测类别 | 真实类别 | IoU | 图像路径 |")
    lines.append("|------|--------|----------|----------|-----|----------|")
    
    for rank, fp in enumerate(top_k_fps, 1):
        pred_cls = fp['pred_class']
        gt_cls = fp['gt_class']
        conf = fp['conf']
        iou = fp['iou']
        img_path = Path(fp['image_path']).name
        
        pred_name = class_names[pred_cls] if pred_cls < len(class_names) else 'Unknown'
        gt_name = class_names[gt_cls] if gt_cls < len(class_names) else 'Unknown'
        
        lines.append(f"| {rank} | {conf:.3f} | {pred_name} ({pred_cls}) | {gt_name} ({gt_cls}) | {iou:.3f} | `{img_path}` |")
    
    lines.append("")
    
    # 3. 分析建议
    lines.append("## 3. 改进建议")
    lines.append("")
    
    if error_pairs:
        top_pair = error_pairs.most_common(1)[0]
        pred_cls, gt_cls = top_pair[0]
        count = top_pair[1]
        pred_name = class_names[pred_cls] if pred_cls < len(class_names) else 'Unknown'
        gt_name = class_names[gt_cls] if gt_cls < len(class_names) else 'Unknown'
        
        lines.append(f"**主要误检模式**: {pred_name} → {gt_name} (出现 {count} 次)")
        lines.append("")
        lines.append("可能原因：")
        lines.append(f"1. 类别 `{pred_name}` 和 `{gt_name}` 在视觉上相似")
        lines.append(f"2. 训练数据中 `{gt_name}` 样本不足")
        lines.append(f"3. 特征提取能力不足，无法区分细微差异")
        lines.append("")
        lines.append("建议措施：")
        lines.append(f"1. 增加 `{gt_name}` 类别的训练样本")
        lines.append(f"2. 使用数据增强增加 `{pred_name}` 和 `{gt_name}` 的区分度")
        lines.append("3. 调整损失函数权重，加大易混淆类别的惩罚")
        lines.append("4. 引入注意力机制，关注类别间的细微差异")
    
    lines.append("")
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ 误检模式分析已保存: {output_path}")


def analyze_occlusion_proxy(occlusion_data, class_names, output_path):
    """
    遮挡/堆叠代理变量分析
    
    使用bbox重叠率和目标密度作为遮挡/堆叠的代理变量，
    分析其对检测性能的影响
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not occlusion_data:
        print("⚠️  无遮挡数据，跳过分析")
        return
    
    # 转换为numpy数组
    overlap_ratios = np.array([d['overlap_ratio'] for d in occlusion_data])
    densities = np.array([d['density'] for d in occlusion_data])
    is_detected = np.array([d['is_detected'] for d in occlusion_data])
    
    # 分桶分析
    overlap_bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
    density_bins = [0, 1, 3, 5, 10, 100]
    
    results = []
    
    # 按重叠率分桶
    for i in range(len(overlap_bins) - 1):
        bin_min = overlap_bins[i]
        bin_max = overlap_bins[i + 1]
        
        mask = (overlap_ratios >= bin_min) & (overlap_ratios < bin_max)
        bin_data = is_detected[mask]
        
        if len(bin_data) > 0:
            recall = np.mean(bin_data)
            count = len(bin_data)
            detected = np.sum(bin_data)
            
            results.append({
                'variable': 'overlap_ratio',
                'bin_min': bin_min,
                'bin_max': bin_max,
                'count': count,
                'detected': detected,
                'recall': recall,
            })
    
    # 按密度分桶
    for i in range(len(density_bins) - 1):
        bin_min = density_bins[i]
        bin_max = density_bins[i + 1]
        
        mask = (densities >= bin_min) & (densities < bin_max)
        bin_data = is_detected[mask]
        
        if len(bin_data) > 0:
            recall = np.mean(bin_data)
            count = len(bin_data)
            detected = np.sum(bin_data)
            
            results.append({
                'variable': 'density',
                'bin_min': bin_min,
                'bin_max': bin_max,
                'count': count,
                'detected': detected,
                'recall': recall,
            })
    
    # 保存CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['variable', 'bin_min', 'bin_max', 'count', 'detected', 'recall'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ 遮挡代理分析已保存: {output_path}")
    
    # 打印简要统计
    print("\n遮挡/堆叠影响分析:")
    print("\n按重叠率分桶:")
    print(f"{'重叠率范围':<20} {'样本数':<10} {'检出数':<10} {'召回率':<10}")
    print("-" * 50)
    for r in results:
        if r['variable'] == 'overlap_ratio':
            bin_range = f"[{r['bin_min']:.1f}, {r['bin_max']:.1f})"
            print(f"{bin_range:<20} {r['count']:<10} {r['detected']:<10} {r['recall']:.3f}")
    
    print("\n按目标密度分桶:")
    print(f"{'密度范围':<20} {'样本数':<10} {'检出数':<10} {'召回率':<10}")
    print("-" * 50)
    for r in results:
        if r['variable'] == 'density':
            bin_range = f"[{r['bin_min']:.1f}, {r['bin_max']:.1f})"
            print(f"{bin_range:<20} {r['count']:<10} {r['detected']:<10} {r['recall']:.3f}")


def main():
    """主函数"""
    args = parse_args()
    
    print("\n" + "="*80)
    print("深度错误分析")
    print("="*80)
    print(f"模型权重: {args.weights}")
    print(f"数据划分: {args.split}")
    print(f"置信度阈值: {args.conf}")
    print(f"IoU匹配阈值: {args.iou_match}")
    
    # 推断实验名称
    weights_path = Path(args.weights)
    if 'runs' in weights_path.parts:
        exp_name = weights_path.parts[weights_path.parts.index('runs') + 1]
        seed_name = weights_path.parts[weights_path.parts.index('runs') + 2]
    else:
        exp_name = "unknown"
        seed_name = "seed0"
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"results/analysis/{exp_name}_{seed_name}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 加载模型
    print("\n" + "="*80)
    print("步骤 1: 加载模型")
    print("="*80)
    model = load_model(args.weights, args.device)
    
    # 获取数据配置
    data_yaml = Path("datasets/data.yaml")
    if not data_yaml.exists():
        print(f"❌ 数据配置文件不存在: {data_yaml}")
        return 1
    
    # 运行推理和分析
    print("\n" + "="*80)
    print("步骤 2: 推理和错误收集")
    print("="*80)
    
    confusion_matrix, per_class_ap, fp_patterns, occlusion_data, class_names = run_inference_and_analyze(
        model, data_yaml, args.split, args.conf, args.iou_match
    )
    
    if confusion_matrix is None:
        return 1
    
    # 生成分析报告
    print("\n" + "="*80)
    print("步骤 3: 生成分析报告")
    print("="*80)
    
    # 1. 混淆矩阵
    plot_confusion_matrix(
        confusion_matrix,
        class_names,
        output_dir / "confusion_matrix.png"
    )
    
    # 2. 每类AP
    save_per_class_ap(
        per_class_ap,
        class_names,
        output_dir / "per_class_ap.csv"
    )
    
    # 3. 误检模式分析
    analyze_fp_patterns(
        fp_patterns,
        class_names,
        output_dir / "fp_top_patterns.md",
        args.top_k
    )
    
    # 4. 遮挡代理分析
    analyze_occlusion_proxy(
        occlusion_data,
        class_names,
        output_dir / "occlusion_proxy_analysis.csv"
    )
    
    print("\n" + "="*80)
    print("✅ 错误分析完成！")
    print("="*80)
    print(f"\n输出文件:")
    print(f"  - 混淆矩阵: {output_dir / 'confusion_matrix.png'}")
    print(f"  - 每类AP: {output_dir / 'per_class_ap.csv'}")
    print(f"  - 误检模式: {output_dir / 'fp_top_patterns.md'}")
    print(f"  - 遮挡分析: {output_dir / 'occlusion_proxy_analysis.csv'}")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
