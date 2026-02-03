"""
数据泄漏检测 (Data Leakage Detection)

该脚本使用感知哈希（Perceptual Hash）算法检测数据集中的近重复图像，
防止训练集和验证集/测试集之间的数据泄漏。

功能：
1. 计算每张图片的感知哈希（使用 average hash 或 difference hash）
2. 检测 train 与 val/test 之间的近重复图像
3. 生成泄漏报告 CSV 文件
4. 提供处理建议

使用方法:
    python scripts/check_leakage.py --dataset datasets/openparts
    python scripts/check_leakage.py --dataset datasets/selfparts
    python scripts/check_leakage.py --all  # 检测所有数据集
    
依赖:
    pip install pillow numpy
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='检测数据集中的数据泄漏')
    parser.add_argument('--dataset', type=str,
                        help='数据集目录（如 datasets/openparts）')
    parser.add_argument('--all', action='store_true',
                        help='检测所有数据集')
    parser.add_argument('--threshold', type=int, default=10,
                        help='汉明距离阈值（0-64，越小越严格），默认10')
    parser.add_argument('--hash-method', type=str, default='average',
                        choices=['average', 'difference'],
                        help='哈希算法：average (快速), difference (平衡)')
    parser.add_argument('--output', type=str, default='results/metadata/leakage_report.csv',
                        help='输出报告文件路径')
    parser.add_argument('--hash-size', type=int, default=8,
                        help='哈希计算时的图片缩放尺寸（8x8 或 16x16）')
    return parser.parse_args()


def average_hash(image_path, hash_size=8):
    """
    计算平均哈希（Average Hash / aHash）
    
    原理：
    1. 缩放图片到 hash_size x hash_size
    2. 转换为灰度图
    3. 计算所有像素的平均值
    4. 每个像素大于平均值则为1，否则为0
    
    Args:
        image_path: 图片路径
        hash_size: 哈希尺寸（默认 8，生成 64 位哈希）
    
    Returns:
        str: 十六进制哈希字符串，失败返回 None
    """
    try:
        img = Image.open(image_path).convert('L')  # 转灰度
        img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img).flatten()
        avg = pixels.mean()
        hash_bits = (pixels > avg).astype(int)
        
        # 转换为十六进制字符串
        hash_str = ''.join(str(b) for b in hash_bits)
        hash_int = int(hash_str, 2)
        return f"{hash_int:0{hash_size*hash_size//4}x}"
    except Exception as e:
        print(f"⚠️  计算哈希失败: {image_path.name} - {e}")
        return None


def difference_hash(image_path, hash_size=8):
    """
    计算差异哈希（Difference Hash / dHash）
    
    原理：
    1. 缩放图片到 (hash_size+1) x hash_size
    2. 转换为灰度图
    3. 比较相邻像素：左边 > 右边则为1，否则为0
    
    优点：对图片的平移和缩放更鲁棒
    
    Args:
        image_path: 图片路径
        hash_size: 哈希尺寸（默认 8）
    
    Returns:
        str: 十六进制哈希字符串，失败返回 None
    """
    try:
        img = Image.open(image_path).convert('L')
        img = img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img)
        
        # 水平方向差异
        diff = pixels[:, 1:] > pixels[:, :-1]
        hash_bits = diff.flatten().astype(int)
        
        # 转换为十六进制
        hash_str = ''.join(str(b) for b in hash_bits)
        hash_int = int(hash_str, 2)
        return f"{hash_int:0{hash_size*hash_size//4}x}"
    except Exception as e:
        print(f"⚠️  计算哈希失败: {image_path.name} - {e}")
        return None


def hamming_distance(hash1, hash2):
    """
    计算两个哈希的汉明距离（Hamming Distance）
    
    汉明距离：两个等长字符串对应位置不同字符的个数
    
    Args:
        hash1, hash2: 十六进制哈希字符串
    
    Returns:
        int: 汉明距离（0 表示完全相同）
    """
    if hash1 is None or hash2 is None:
        return float('inf')
    
    # 转换为二进制并计算不同位的数量
    try:
        int1 = int(hash1, 16)
        int2 = int(hash2, 16)
        xor = int1 ^ int2
        return bin(xor).count('1')
    except:
        return float('inf')


def compute_hashes(images_dir, hash_method='average', hash_size=8):
    """
    计算指定目录所有图片的哈希
    
    Args:
        images_dir: 图片目录
        hash_method: 哈希算法
        hash_size: 哈希尺寸
    
    Returns:
        dict: {image_path: hash_value}
    """
    if not images_dir.exists():
        return {}
    
    # 选择哈希函数
    hash_func = average_hash if hash_method == 'average' else difference_hash
    
    # 获取所有图片（使用set去重，避免Windows下大小写导致重复）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = set()  # 使用set自动去重
    for ext in image_extensions:
        image_files.update(images_dir.glob(f'*{ext}'))
        image_files.update(images_dir.glob(f'*{ext.upper()}'))
    
    hashes = {}
    for img_path in image_files:
        img_hash = hash_func(img_path, hash_size)
        if img_hash:
            hashes[img_path] = img_hash
    
    return hashes


def find_duplicates(train_hashes, other_hashes, other_split, threshold=10):
    """
    查找训练集和其他集合之间的近重复图像
    
    Args:
        train_hashes: 训练集哈希字典
        other_hashes: 其他集合（val/test）哈希字典
        other_split: 其他集合的名称（'val' 或 'test'）
        threshold: 汉明距离阈值
    
    Returns:
        list: [(train_img, other_img, distance), ...]
    """
    duplicates = []
    
    for train_path, train_hash in train_hashes.items():
        for other_path, other_hash in other_hashes.items():
            distance = hamming_distance(train_hash, other_hash)
            
            if distance <= threshold:
                duplicates.append({
                    'dataset': train_path.parts[-5],  # 数据集名称
                    'train_image': train_path.name,
                    'train_path': str(train_path),
                    'other_image': other_path.name,
                    'other_path': str(other_path),
                    'other_split': other_split,
                    'hamming_distance': distance,
                    'similarity': f"{(1 - distance / 64) * 100:.1f}%"
                })
    
    return duplicates


def generate_report(all_duplicates, output_path, threshold, hash_method):
    """
    生成数据泄漏报告
    
    Args:
        all_duplicates: 所有数据集的重复图像列表
        output_path: 输出文件路径
        threshold: 使用的阈值
        hash_method: 使用的哈希方法
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入 CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if all_duplicates:
            fieldnames = all_duplicates[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_duplicates)
        else:
            # 即使没有重复，也创建空文件（带表头）
            writer = csv.DictWriter(f, fieldnames=[
                'dataset', 'train_image', 'train_path', 'other_image', 
                'other_path', 'other_split', 'hamming_distance', 'similarity'
            ])
            writer.writeheader()
    
    # 添加建议处理策略（作为注释）
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write('\n# ===== 数据泄漏检测报告说明 =====\n')
        f.write(f'# 检测方法: {hash_method} hash\n')
        f.write(f'# 汉明距离阈值: {threshold} (0-64，越小越严格)\n')
        f.write(f'# 检测到的近重复图像对数: {len(all_duplicates)}\n')
        f.write('#\n')
        f.write('# 【建议处理策略】\n')
        f.write('#\n')
        f.write('# 1. 完全相同（距离=0）：\n')
        f.write('#    - 这是严重的数据泄漏，必须处理\n')
        f.write('#    - 建议：从验证集/测试集中删除重复图像\n')
        f.write('#\n')
        f.write('# 2. 高度相似（距离 1-5）：\n')
        f.write('#    - 可能是同一场景的不同帧或轻微变化\n')
        f.write('#    - 建议：手动检查，确认后删除或移动\n')
        f.write('#\n')
        f.write('# 3. 中度相似（距离 6-10）：\n')
        f.write('#    - 可能是相似场景或目标\n')
        f.write('#    - 建议：手动抽查，评估是否影响实验公平性\n')
        f.write('#\n')
        f.write('# 4. 低度相似（距离 >10）：\n')
        f.write('#    - 通常是误报或偶然相似\n')
        f.write('#    - 建议：可以保留，但需记录在实验报告中\n')
        f.write('#\n')
        f.write('# 【处理步骤】\n')
        f.write('#\n')
        f.write('# 方案A（推荐）：从验证集/测试集中删除\n')
        f.write('#   1. 备份原始数据集\n')
        f.write('#   2. 根据 other_path 列删除重复图像及其标签\n')
        f.write('#   3. 重新运行统计脚本更新 dataset_stats.json\n')
        f.write('#   4. 重新训练模型\n')
        f.write('#\n')
        f.write('# 方案B（替代）：从训练集中删除\n')
        f.write('#   1. 如果训练集样本充足，可以从训练集删除\n')
        f.write('#   2. 根据 train_path 列删除重复图像及其标签\n')
        f.write('#   3. 重新运行统计脚本\n')
        f.write('#\n')
        f.write('# 【注意事项】\n')
        f.write('#\n')
        f.write('# - 删除操作不可逆，务必先备份\n')
        f.write('# - 感知哈希可能存在误报，建议人工复核距离=0的情况\n')
        f.write('# - 处理后重新运行 check_leakage.py 确认\n')
        f.write('# - 在实验报告中记录所有处理步骤\n')


def check_dataset_leakage(dataset_dir, args):
    """检测单个数据集的泄漏"""
    dataset_path = Path(dataset_dir)
    dataset_name = dataset_path.name
    
    print(f"\n{'='*80}")
    print(f"检测数据集: {dataset_name}")
    print(f"{'='*80}")
    
    # 计算各个划分的哈希
    print(f"📊 计算图像哈希...")
    
    train_dir = dataset_path / 'images' / 'train'
    val_dir = dataset_path / 'images' / 'val'
    test_dir = dataset_path / 'images' / 'test'
    
    train_hashes = compute_hashes(train_dir, args.hash_method, args.hash_size)
    val_hashes = compute_hashes(val_dir, args.hash_method, args.hash_size)
    test_hashes = compute_hashes(test_dir, args.hash_method, args.hash_size)
    
    print(f"  train: {len(train_hashes)} 张")
    print(f"  val:   {len(val_hashes)} 张")
    print(f"  test:  {len(test_hashes)} 张")
    
    if len(train_hashes) == 0:
        print(f"⚠️  训练集为空，跳过检测")
        return []
    
    # 检测 train vs val
    duplicates = []
    
    if len(val_hashes) > 0:
        print(f"\n🔍 检测 train vs val...")
        train_val_dups = find_duplicates(train_hashes, val_hashes, 'val', args.threshold)
        duplicates.extend(train_val_dups)
        print(f"  发现 {len(train_val_dups)} 对近重复图像")
    
    # 检测 train vs test
    if len(test_hashes) > 0:
        print(f"\n🔍 检测 train vs test...")
        train_test_dups = find_duplicates(train_hashes, test_hashes, 'test', args.threshold)
        duplicates.extend(train_test_dups)
        print(f"  发现 {len(train_test_dups)} 对近重复图像")
    
    # 打印摘要
    if duplicates:
        identical = sum(1 for d in duplicates if d['hamming_distance'] == 0)
        high_sim = sum(1 for d in duplicates if 1 <= d['hamming_distance'] <= 5)
        medium_sim = sum(1 for d in duplicates if 6 <= d['hamming_distance'] <= 10)
        
        print(f"\n⚠️  数据集 {dataset_name} 发现 {len(duplicates)} 对近重复图像")
        print(f"  完全相同 (距离=0):   {identical:3d} 对")
        print(f"  高度相似 (距离1-5):  {high_sim:3d} 对")
        print(f"  中度相似 (距离6-10): {medium_sim:3d} 对")
    else:
        print(f"\n✅ 数据集 {dataset_name} 未发现数据泄漏")
    
    return duplicates


def main():
    """主函数"""
    args = parse_args()
    
    # 确定要检测的数据集列表
    datasets_to_check = []
    
    if args.all:
        # 自动发现所有数据集
        datasets_root = Path('datasets')
        if datasets_root.exists():
            for subdir in datasets_root.iterdir():
                if subdir.is_dir():
                    yaml_path = subdir / 'data.yaml'
                    if yaml_path.exists():
                        datasets_to_check.append(subdir)
        
        if not datasets_to_check:
            print("❌ 未找到任何数据集")
            return
    
    elif args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"❌ 数据集目录不存在: {dataset_path}")
            return
        datasets_to_check.append(dataset_path)
    
    else:
        print("❌ 请指定 --dataset 或 --all")
        return
    
    print("=" * 80)
    print("数据泄漏检测工具 (Data Leakage Checker)")
    print("=" * 80)
    print(f"哈希方法: {args.hash_method}")
    print(f"哈希尺寸: {args.hash_size}x{args.hash_size}")
    print(f"汉明距离阈值: {args.threshold}")
    print(f"输出文件: {args.output}")
    print(f"\n将检测 {len(datasets_to_check)} 个数据集:")
    for ds in datasets_to_check:
        print(f"  - {ds}")
    
    # 检测所有数据集
    all_duplicates = []
    
    for dataset_path in datasets_to_check:
        duplicates = check_dataset_leakage(dataset_path, args)
        all_duplicates.extend(duplicates)
    
    # 生成报告
    print(f"\n\n📝 生成报告...")
    generate_report(all_duplicates, args.output, args.threshold, args.hash_method)
    print(f"   ✓ {args.output}")
    
    # 打印总结
    print("\n" + "=" * 80)
    if all_duplicates:
        # 按数据集分组统计
        dataset_summary = defaultdict(int)
        for dup in all_duplicates:
            dataset_summary[dup['dataset']] += 1
        
        print(f"⚠️  总计发现 {len(all_duplicates)} 对近重复图像")
        print(f"\n   各数据集统计:")
        for dataset, count in dataset_summary.items():
            print(f"     {dataset}: {count} 对")
        
        # 按严重程度统计
        identical = sum(1 for d in all_duplicates if d['hamming_distance'] == 0)
        high_sim = sum(1 for d in all_duplicates if 1 <= d['hamming_distance'] <= 5)
        
        print(f"\n   严重程度分布:")
        print(f"     完全相同 (距离=0):   {identical} 对 {'⚠️  严重' if identical > 0 else ''}")
        print(f"     高度相似 (距离1-5):  {high_sim} 对 {'⚠️  需处理' if high_sim > 0 else ''}")
        
        if identical > 0:
            print(f"\n   🔴 发现完全相同的图像，这是严重的数据泄漏！")
            print(f"      请立即查看报告并处理: {args.output}")
    else:
        print("✅ 未发现任何数据泄漏！所有数据集划分良好。")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
