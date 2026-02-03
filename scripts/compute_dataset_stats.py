"""
数据集统计工具 (Dataset Statistics)

该脚本统计 YOLO 格式数据集的详细信息，包括：
1. 每个类别的样本数（含该类别的图片数）
2. 每个类别的标注框数
3. 标注框面积分布（small/medium/large，按 COCO 标准）
4. 数据集整体统计信息

使用方法:
    python scripts/compute_dataset_stats.py --dataset datasets/openparts
    python scripts/compute_dataset_stats.py --dataset datasets/selfparts
    python scripts/compute_dataset_stats.py --all  # 统计所有数据集
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='统计 YOLO 数据集信息')
    parser.add_argument('--dataset', type=str, 
                        help='数据集目录路径（如 datasets/openparts）')
    parser.add_argument('--all', action='store_true',
                        help='统计所有数据集')
    parser.add_argument('--output', type=str, 
                        default='results/metadata/dataset_stats.json',
                        help='输出文件路径')
    parser.add_argument('--image-size', type=int, default=640,
                        help='假设的图片尺寸（用于计算面积，默认640）')
    return parser.parse_args()


def load_yaml_config(yaml_path):
    """加载 YAML 配置文件"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parse_label_file(label_path):
    """
    解析 YOLO 格式标签文件
    
    Args:
        label_path: 标签文件路径
    
    Returns:
        list: [(class_id, x_center, y_center, width, height), ...]
    """
    bboxes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    bboxes.append((class_id, x, y, w, h))
    except Exception as e:
        print(f"⚠️  解析标签文件失败: {label_path} - {e}")
    
    return bboxes


def compute_dataset_stats(dataset_dir, config, image_size=640):
    """
    计算单个数据集的统计信息
    
    Args:
        dataset_dir: 数据集根目录
        config: YAML 配置
        image_size: 假设的图片尺寸
    
    Returns:
        dict: 统计信息
    """
    dataset_path = Path(dataset_dir)
    class_names = config.get('names', {})
    
    # 转换为列表格式
    if isinstance(class_names, dict):
        class_names_list = [class_names[i] for i in sorted(class_names.keys())]
    else:
        class_names_list = class_names
    
    stats = {
        'dataset_name': dataset_path.name,
        'dataset_path': str(dataset_path),
        'num_classes': config.get('nc', len(class_names_list)),
        'class_names': class_names_list,
        'splits': {},
        'class_distribution': {},
        'bbox_size_distribution': {
            'small': 0,   # area < 32^2
            'medium': 0,  # 32^2 <= area < 96^2
            'large': 0    # area >= 96^2
        },
        'total_images': 0,
        'total_bboxes': 0,
    }
    
    # 初始化类别统计
    for class_id, class_name in enumerate(class_names_list):
        stats['class_distribution'][class_name] = {
            'class_id': class_id,
            'images_count': 0,      # 包含该类别的图片数
            'bboxes_count': 0,      # 该类别的标注框总数
            'avg_boxes_per_image': 0.0,
        }
    
    # 遍历各个划分
    splits_to_check = ['train', 'val', 'test']
    
    for split in splits_to_check:
        images_dir = dataset_path / 'images' / split
        labels_dir = dataset_path / 'labels' / split
        
        if not images_dir.exists():
            continue
        
        # 获取所有图片文件（使用set去重，避免Windows下大小写导致重复）
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = set()  # 使用set自动去重
        for ext in image_extensions:
            image_files.update(images_dir.glob(f'*{ext}'))
            image_files.update(images_dir.glob(f'*{ext.upper()}'))
        
        image_files = list(image_files)  # 转回列表
        
        split_stats = {
            'num_images': len(image_files),
            'num_bboxes': 0,
            'class_counts': defaultdict(int),
            'class_image_counts': defaultdict(int),  # 记录每个类别出现在多少张图片中
        }
        
        # 统计每个图片的标签
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
            
            bboxes = parse_label_file(label_path)
            split_stats['num_bboxes'] += len(bboxes)
            
            # 记录该图片中出现的类别（去重）
            classes_in_image = set()
            
            for class_id, x, y, w, h in bboxes:
                split_stats['class_counts'][class_id] += 1
                classes_in_image.add(class_id)
                
                # 计算bbox面积（归一化坐标，需乘以图片尺寸的平方）
                area_pixels = w * h * image_size * image_size
                
                if area_pixels < 32 * 32:
                    stats['bbox_size_distribution']['small'] += 1
                elif area_pixels < 96 * 96:
                    stats['bbox_size_distribution']['medium'] += 1
                else:
                    stats['bbox_size_distribution']['large'] += 1
            
            # 更新类别的图片计数
            for cls_id in classes_in_image:
                split_stats['class_image_counts'][cls_id] += 1
        
        # 保存划分统计
        stats['splits'][split] = {
            'num_images': split_stats['num_images'],
            'num_bboxes': split_stats['num_bboxes'],
            'avg_boxes_per_image': split_stats['num_bboxes'] / split_stats['num_images'] if split_stats['num_images'] > 0 else 0,
            'class_counts': dict(split_stats['class_counts']),
            'class_image_counts': dict(split_stats['class_image_counts']),
        }
        
        stats['total_images'] += split_stats['num_images']
        stats['total_bboxes'] += split_stats['num_bboxes']
    
    # 汇总类别统计（跨所有划分）
    for split_name, split_data in stats['splits'].items():
        for cls_id, count in split_data['class_counts'].items():
            if cls_id < len(class_names_list):
                class_name = class_names_list[cls_id]
                stats['class_distribution'][class_name]['bboxes_count'] += count
        
        for cls_id, count in split_data['class_image_counts'].items():
            if cls_id < len(class_names_list):
                class_name = class_names_list[cls_id]
                stats['class_distribution'][class_name]['images_count'] += count
    
    # 计算平均值
    for class_name, info in stats['class_distribution'].items():
        if info['images_count'] > 0:
            info['avg_boxes_per_image'] = round(info['bboxes_count'] / info['images_count'], 2)
    
    return stats


def print_dataset_stats(stats):
    """打印数据集统计信息到控制台"""
    print(f"\n{'='*80}")
    print(f"数据集: {stats['dataset_name']}")
    print(f"{'='*80}")
    
    print(f"\n📊 整体统计:")
    print(f"  总图片数: {stats['total_images']}")
    print(f"  总标注框数: {stats['total_bboxes']}")
    print(f"  类别数: {stats['num_classes']}")
    if stats['total_images'] > 0:
        print(f"  平均每张图: {stats['total_bboxes'] / stats['total_images']:.2f} 个框")
    
    print(f"\n📂 各划分统计:")
    for split in ['train', 'val', 'test']:
        if split in stats['splits']:
            split_info = stats['splits'][split]
            print(f"  {split:5s}: {split_info['num_images']:4d} 张图片, "
                  f"{split_info['num_bboxes']:5d} 个框, "
                  f"平均 {split_info['avg_boxes_per_image']:.2f} 框/图")
    
    print(f"\n📦 目标尺寸分布 (COCO标准):")
    size_dist = stats['bbox_size_distribution']
    total_boxes = size_dist['small'] + size_dist['medium'] + size_dist['large']
    if total_boxes > 0:
        print(f"  Small  (<32²px):    {size_dist['small']:5d} ({size_dist['small']/total_boxes*100:5.1f}%)")
        print(f"  Medium (32²-96²px): {size_dist['medium']:5d} ({size_dist['medium']/total_boxes*100:5.1f}%)")
        print(f"  Large  (≥96²px):    {size_dist['large']:5d} ({size_dist['large']/total_boxes*100:5.1f}%)")
    
    print(f"\n🏷️  类别分布:")
    print(f"  {'类别名称':<30} {'图片数':<10} {'框数':<10} {'平均框/图':<10}")
    print(f"  {'-'*65}")
    
    for class_name, info in stats['class_distribution'].items():
        if info['bboxes_count'] > 0:
            print(f"  {class_name:<30} {info['images_count']:<10} "
                  f"{info['bboxes_count']:<10} {info['avg_boxes_per_image']:<10.2f}")


def main():
    """主函数"""
    args = parse_args()
    
    # 确定要处理的数据集列表
    datasets_to_process = []
    
    if args.all:
        # 自动发现所有数据集
        datasets_root = Path('datasets')
        if datasets_root.exists():
            for subdir in datasets_root.iterdir():
                if subdir.is_dir():
                    yaml_path = subdir / 'data.yaml'
                    if yaml_path.exists():
                        datasets_to_process.append(subdir)
        
        if not datasets_to_process:
            print("❌ 未找到任何数据集")
            return
    
    elif args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"❌ 数据集目录不存在: {dataset_path}")
            return
        
        yaml_path = dataset_path / 'data.yaml'
        if not yaml_path.exists():
            print(f"❌ 配置文件不存在: {yaml_path}")
            return
        
        datasets_to_process.append(dataset_path)
    
    else:
        print("❌ 请指定 --dataset 或 --all")
        return
    
    print("=" * 80)
    print("数据集统计工具 (Dataset Statistics)")
    print("=" * 80)
    print(f"\n将统计 {len(datasets_to_process)} 个数据集:")
    for ds in datasets_to_process:
        print(f"  - {ds}")
    
    # 统计所有数据集
    all_stats = {}
    
    for dataset_path in datasets_to_process:
        print(f"\n\n🔍 正在统计: {dataset_path}...")
        
        # 加载配置
        yaml_path = dataset_path / 'data.yaml'
        config = load_yaml_config(yaml_path)
        
        # 计算统计信息
        stats = compute_dataset_stats(dataset_path, config, args.image_size)
        
        # 打印统计信息
        print_dataset_stats(stats)
        
        # 保存到结果中
        all_stats[dataset_path.name] = stats
    
    # 保存到 JSON 文件
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n{'='*80}")
    print(f"✅ 统计完成！")
    print(f"   输出文件: {output_path}")
    print(f"   文件大小: {output_path.stat().st_size} bytes")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
