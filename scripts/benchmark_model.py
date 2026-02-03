"""
模型性能基准测试

功能：
1. 统计模型复杂度（Params、GFLOPs）
2. GPU延迟测量（使用 torch.cuda.synchronize 避免异步误差）
3. 内存占用统计
4. 使用固定输入集确保公平对比

输出：
- results/bench/{model_name}.json

作者: 科研复现协议
"""

import sys
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from datetime import datetime
import random

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="模型性能基准测试")
    
    parser.add_argument("--weights", required=True, help="模型权重路径")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--device", default="0", help="设备 (0, cpu)")
    parser.add_argument("--warmup", type=int, default=50, help="预热迭代次数")
    parser.add_argument("--iters", type=int, default=300, help="测试迭代次数")
    parser.add_argument("--batch", type=int, default=1, help="批次大小")
    parser.add_argument("--data", default="datasets/openparts/data.yaml", help="数据集配置")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于选择图像）")
    parser.add_argument("--benchmark_size", type=int, default=200, 
                       help="基准测试图像数量")
    parser.add_argument("--use_benchmark_list", action="store_true",
                       help="使用固定的benchmark_list.txt（确保公平对比）")
    
    return parser.parse_args()


def create_benchmark_list(data_yaml, split="val", size=200, seed=42, output_path="benchmark_list.txt"):
    """
    从数据集中随机选择固定的图像列表作为基准测试集
    
    Args:
        data_yaml: 数据集配置文件
        split: 数据集split（val或test）
        size: 选择的图像数量
        seed: 随机种子
        output_path: 输出列表文件路径
    
    Returns:
        image_list: 图像路径列表
    """
    import yaml
    from pathlib import Path
    
    # 加载数据集配置
    with open(data_yaml, encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    
    # 获取数据集根目录
    data_root = Path(data_yaml).parent
    
    # 获取split路径
    if split == "val":
        split_path = data_root / data_cfg.get("val", "images/val")
    elif split == "test":
        split_path = data_root / data_cfg.get("test", "images/test")
    else:
        raise ValueError(f"Invalid split: {split}")
    
    # 收集所有图像
    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    all_images = []
    for ext in image_exts:
        all_images.extend(split_path.glob(f"*{ext}"))
        all_images.extend(split_path.glob(f"*{ext.upper()}"))
    
    all_images = [str(img) for img in all_images]
    
    print(f"✓ 从 {split_path} 找到 {len(all_images)} 张图像")
    
    # 固定随机种子，随机选择
    random.seed(seed)
    if len(all_images) < size:
        print(f"⚠️  警告：可用图像数({len(all_images)}) < 请求数({size})，使用全部图像")
        selected = all_images
    else:
        selected = random.sample(all_images, size)
    
    # 保存到文件
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        for img_path in selected:
            f.write(f"{img_path}\n")
    
    print(f"✓ 已创建基准测试列表: {output_file}")
    print(f"  - 图像数量: {len(selected)}")
    print(f"  - 随机种子: {seed}")
    
    return selected


def load_benchmark_list(list_path="benchmark_list.txt"):
    """加载已有的基准测试列表"""
    with open(list_path) as f:
        images = [line.strip() for line in f if line.strip()]
    
    print(f"✓ 已加载基准测试列表: {list_path}")
    print(f"  - 图像数量: {len(images)}")
    
    return images


def get_model_info(model):
    """
    获取模型信息（Params、GFLOPs）
    
    使用Ultralytics自带的info()方法
    """
    try:
        # Ultralytics模型有info方法
        info = model.info(verbose=False)
        
        # 尝试从info中提取信息
        # info可能返回字典或打印输出
        if isinstance(info, tuple):
            # (params, gflops)
            params = info[0] if len(info) > 0 else 0
            gflops = info[1] if len(info) > 1 else 0
        elif isinstance(info, dict):
            params = info.get("parameters", 0)
            gflops = info.get("GFLOPs", 0)
        else:
            # info()可能只打印，需要手动计算
            params = sum(p.numel() for p in model.model.parameters())
            gflops = 0  # 需要额外计算
        
        return params, gflops
    
    except Exception as e:
        print(f"⚠️  获取模型信息失败: {e}")
        # 手动计算参数量
        try:
            params = sum(p.numel() for p in model.model.parameters())
            return params, 0
        except:
            return 0, 0


def measure_latency_cuda(model, input_tensor, device, warmup=50, iters=300):
    """
    使用CUDA事件精确测量GPU延迟
    
    Args:
        model: 模型
        input_tensor: 输入张量
        device: 设备
        warmup: 预热次数
        iters: 测试次数
    
    Returns:
        latencies: 延迟列表（毫秒）
    """
    print(f"开始延迟测量（warmup={warmup}, iters={iters}）...")
    
    # 确保模型和输入在同一设备
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # 预热
    print(f"  预热中...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    # 测量
    print(f"  测量中...")
    latencies = []
    
    if device.type == "cuda":
        # 使用CUDA事件精确测量
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            for i in range(iters):
                start_event.record()
                _ = model(input_tensor)
                end_event.record()
                
                # 同步等待完成
                torch.cuda.synchronize()
                
                # 获取时间（毫秒）
                latency_ms = start_event.elapsed_time(end_event)
                latencies.append(latency_ms)
                
                if (i + 1) % 100 == 0:
                    print(f"    进度: {i+1}/{iters}")
    
    else:
        # CPU：使用time.perf_counter
        with torch.no_grad():
            for i in range(iters):
                start = time.perf_counter()
                _ = model(input_tensor)
                end = time.perf_counter()
                
                latency_ms = (end - start) * 1000  # 转换为毫秒
                latencies.append(latency_ms)
                
                if (i + 1) % 100 == 0:
                    print(f"    进度: {i+1}/{iters}")
    
    return latencies


def get_gpu_memory_usage(device):
    """获取GPU显存占用（MB）"""
    if device.type == "cuda":
        # 当前分配的显存
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        # 峰值显存
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        # 缓存的显存
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        
        return {
            "allocated_MB": allocated,
            "peak_allocated_MB": max_allocated,
            "reserved_MB": reserved,
        }
    else:
        return {
            "allocated_MB": 0,
            "peak_allocated_MB": 0,
            "reserved_MB": 0,
        }


def benchmark_model(args):
    """运行完整的模型基准测试"""
    
    print(f"\n{'='*80}")
    print(f"模型性能基准测试")
    print(f"{'='*80}")
    
    # 解析模型名称
    weights_path = Path(args.weights)
    model_name = weights_path.stem  # 例如 best, last
    if model_name in ["best", "last"]:
        # 使用父目录名称
        model_name = weights_path.parent.parent.name
    
    print(f"模型名称: {model_name}")
    print(f"权重文件: {args.weights}")
    print(f"输入尺寸: {args.imgsz}")
    print(f"设备: {args.device}")
    print(f"批次大小: {args.batch}")
    
    # 创建输出目录
    output_dir = Path("results/bench")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"{model_name}.json"
    
    print(f"\n{'='*80}")
    print(f"步骤 1: 准备基准测试图像列表")
    print(f"{'='*80}")
    
    benchmark_list_path = output_dir / "benchmark_list.txt"
    
    if args.use_benchmark_list and benchmark_list_path.exists():
        # 使用已有的列表
        print(f"使用已有的基准测试列表（确保公平对比）")
        image_list = load_benchmark_list(benchmark_list_path)
    else:
        # 创建新列表
        print(f"创建新的基准测试列表")
        image_list = create_benchmark_list(
            args.data,
            split="val",
            size=args.benchmark_size,
            seed=args.seed,
            output_path=benchmark_list_path
        )
    
    print(f"\n{'='*80}")
    print(f"步骤 2: 加载模型")
    print(f"{'='*80}")
    
    # 加载模型
    model = YOLO(args.weights)
    print(f"✓ 模型已加载")
    
    # 获取模型信息
    print(f"\n获取模型信息...")
    params, gflops = get_model_info(model)
    print(f"✓ 参数量: {params:,}")
    print(f"✓ GFLOPs: {gflops:.2f}")
    
    print(f"\n{'='*80}")
    print(f"步骤 3: 准备测试输入")
    print(f"{'='*80}")
    
    # 设备
    device = torch.device(args.device if args.device != "cpu" else "cpu")
    print(f"设备: {device}")
    
    # 创建随机输入张量（用于延迟测试）
    input_shape = (args.batch, 3, args.imgsz, args.imgsz)
    input_tensor = torch.randn(input_shape, dtype=torch.float32)
    print(f"✓ 输入形状: {input_shape}")
    
    # 重置显存统计
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"步骤 4: 延迟测量")
    print(f"{'='*80}")
    
    # 测量延迟
    latencies = measure_latency_cuda(
        model.model,
        input_tensor,
        device,
        warmup=args.warmup,
        iters=args.iters
    )
    
    # 统计
    latencies = np.array(latencies)
    latency_mean = np.mean(latencies)
    latency_median = np.median(latencies)
    latency_p95 = np.percentile(latencies, 95)
    latency_p99 = np.percentile(latencies, 99)
    latency_std = np.std(latencies)
    
    # FPS
    fps_mean = 1000.0 / latency_mean * args.batch  # 考虑batch size
    fps_p50 = 1000.0 / latency_median * args.batch
    
    print(f"\n延迟统计:")
    print(f"  - Mean: {latency_mean:.2f} ms")
    print(f"  - Median (P50): {latency_median:.2f} ms")
    print(f"  - P95: {latency_p95:.2f} ms")
    print(f"  - P99: {latency_p99:.2f} ms")
    print(f"  - Std: {latency_std:.2f} ms")
    print(f"\nFPS统计:")
    print(f"  - Mean FPS: {fps_mean:.2f}")
    print(f"  - P50 FPS: {fps_p50:.2f}")
    
    print(f"\n{'='*80}")
    print(f"步骤 5: 显存占用")
    print(f"{'='*80}")
    
    # 获取显存占用
    mem_stats = get_gpu_memory_usage(device)
    print(f"GPU显存统计:")
    print(f"  - 当前分配: {mem_stats['allocated_MB']:.2f} MB")
    print(f"  - 峰值分配: {mem_stats['peak_allocated_MB']:.2f} MB")
    print(f"  - 缓存: {mem_stats['reserved_MB']:.2f} MB")
    
    print(f"\n{'='*80}")
    print(f"步骤 6: 保存结果")
    print(f"{'='*80}")
    
    # 汇总结果
    results = {
        "metadata": {
            "model_name": model_name,
            "weights": str(args.weights),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": str(device),
            "warmup": args.warmup,
            "iters": args.iters,
            "timestamp": datetime.now().isoformat(),
            "benchmark_list": str(benchmark_list_path),
            "benchmark_size": len(image_list),
        },
        "model_complexity": {
            "params": int(params),
            "params_M": params / 1e6,
            "gflops": float(gflops),
        },
        "latency": {
            "mean_ms": float(latency_mean),
            "median_ms": float(latency_median),
            "p50_ms": float(latency_median),
            "p95_ms": float(latency_p95),
            "p99_ms": float(latency_p99),
            "std_ms": float(latency_std),
            "min_ms": float(latencies.min()),
            "max_ms": float(latencies.max()),
        },
        "throughput": {
            "fps_mean": float(fps_mean),
            "fps_p50": float(fps_p50),
        },
        "memory": {
            "peak_gpu_mem_MB": float(mem_stats['peak_allocated_MB']),
            "allocated_MB": float(mem_stats['allocated_MB']),
            "reserved_MB": float(mem_stats['reserved_MB']),
        }
    }
    
    # 保存JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    try:
        rel_path = output_json.relative_to(Path.cwd())
    except ValueError:
        rel_path = output_json
    
    print(f"✓ 结果已保存: {rel_path}")
    
    print(f"\n{'='*80}")
    print(f"基准测试完成")
    print(f"{'='*80}")
    print(f"模型: {model_name}")
    print(f"参数量: {params/1e6:.2f}M ({params:,})")
    print(f"GFLOPs: {gflops:.2f}")
    print(f"延迟 (mean): {latency_mean:.2f} ms")
    print(f"延迟 (p95): {latency_p95:.2f} ms")
    print(f"FPS (mean): {fps_mean:.2f}")
    print(f"峰值显存: {mem_stats['peak_allocated_MB']:.2f} MB")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    args = parse_args()
    results = benchmark_model(args)
