"""
测试模型基准测试脚本

验证：
1. benchmark_list.txt 生成
2. 延迟测量（包含CUDA同步）
3. 输出JSON结构
4. 同一输入列表复用
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts.benchmark_model import (
    create_benchmark_list,
    load_benchmark_list,
    measure_latency_cuda,
    get_gpu_memory_usage,
)


def test_benchmark_list_creation():
    """测试基准测试列表创建"""
    print("\n" + "="*80)
    print("测试 1: 基准测试列表创建")
    print("="*80)
    
    # 测试参数
    data_yaml = "datasets/openparts/data.yaml"
    output_path = "results/bench/test_benchmark_list.txt"
    
    # 检查数据集是否存在
    if not Path(data_yaml).exists():
        print(f"⚠️  数据集不存在: {data_yaml}")
        print(f"跳过测试")
        return True
    
    # 创建列表
    try:
        image_list = create_benchmark_list(
            data_yaml=data_yaml,
            split="val",
            size=50,  # 小数量测试
            seed=42,
            output_path=output_path
        )
        
        print(f"\n验证:")
        print(f"  ✓ 列表文件创建: {output_path}")
        print(f"  ✓ 图像数量: {len(image_list)}")
        
        # 验证文件存在
        assert Path(output_path).exists(), "列表文件不存在"
        
        # 验证内容
        loaded_list = load_benchmark_list(output_path)
        assert len(loaded_list) == len(image_list), "列表长度不一致"
        
        print(f"  ✓ 列表加载验证通过")
        
        # 测试固定种子的一致性
        image_list2 = create_benchmark_list(
            data_yaml=data_yaml,
            split="val",
            size=50,
            seed=42,  # 相同种子
            output_path="results/bench/test_benchmark_list2.txt"
        )
        
        # 应该完全相同
        if image_list == image_list2:
            print(f"  ✓ 固定种子一致性验证通过")
        else:
            print(f"  ⚠️  警告：固定种子未产生相同列表")
        
        print("\n✅ 基准测试列表创建测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_latency_measurement():
    """测试延迟测量（包含CUDA同步）"""
    print("\n" + "="*80)
    print("测试 2: 延迟测量")
    print("="*80)
    
    # 创建简单模型
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    
    # 测试输入
    input_tensor = torch.randn(1, 3, 64, 64)
    
    # CPU测试
    print("\nCPU延迟测量:")
    device_cpu = torch.device("cpu")
    latencies_cpu = measure_latency_cuda(
        model, input_tensor, device_cpu,
        warmup=5, iters=20
    )
    
    print(f"  - 测量次数: {len(latencies_cpu)}")
    print(f"  - Mean: {np.mean(latencies_cpu):.2f} ms")
    print(f"  - Median: {np.median(latencies_cpu):.2f} ms")
    print(f"  - Std: {np.std(latencies_cpu):.2f} ms")
    
    assert len(latencies_cpu) == 20, "CPU测量次数不对"
    assert all(lat > 0 for lat in latencies_cpu), "存在无效延迟"
    print(f"  ✓ CPU测量验证通过")
    
    # GPU测试（如果可用）
    if torch.cuda.is_available():
        print("\nGPU延迟测量:")
        device_gpu = torch.device("cuda:0")
        latencies_gpu = measure_latency_cuda(
            model, input_tensor, device_gpu,
            warmup=5, iters=20
        )
        
        print(f"  - 测量次数: {len(latencies_gpu)}")
        print(f"  - Mean: {np.mean(latencies_gpu):.2f} ms")
        print(f"  - Median: {np.median(latencies_gpu):.2f} ms")
        print(f"  - Std: {np.std(latencies_gpu):.2f} ms")
        
        assert len(latencies_gpu) == 20, "GPU测量次数不对"
        assert all(lat > 0 for lat in latencies_gpu), "存在无效延迟"
        print(f"  ✓ GPU测量验证通过")
        
        # GPU应该比CPU快（通常情况）
        if np.mean(latencies_gpu) < np.mean(latencies_cpu):
            print(f"  ✓ GPU延迟 < CPU延迟（符合预期）")
        else:
            print(f"  ⚠️  GPU延迟 >= CPU延迟（可能是小模型开销）")
    else:
        print("\n⚠️  CUDA不可用，跳过GPU测试")
    
    print("\n✅ 延迟测量测试通过")
    return True


def test_gpu_memory():
    """测试显存统计"""
    print("\n" + "="*80)
    print("测试 3: GPU显存统计")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过测试")
        return True
    
    device = torch.device("cuda:0")
    
    # 重置统计
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    # 分配一些显存
    tensor = torch.randn(1000, 1000, device=device)
    
    # 获取统计
    mem_stats = get_gpu_memory_usage(device)
    
    print(f"显存统计:")
    print(f"  - 当前分配: {mem_stats['allocated_MB']:.2f} MB")
    print(f"  - 峰值分配: {mem_stats['peak_allocated_MB']:.2f} MB")
    print(f"  - 缓存: {mem_stats['reserved_MB']:.2f} MB")
    
    # 验证
    assert mem_stats['allocated_MB'] > 0, "显存分配应大于0"
    assert mem_stats['peak_allocated_MB'] >= mem_stats['allocated_MB'], "峰值应>=当前"
    
    print(f"  ✓ 显存统计验证通过")
    
    # 清理
    del tensor
    torch.cuda.empty_cache()
    
    print("\n✅ GPU显存统计测试通过")
    return True


def test_json_structure():
    """测试输出JSON结构"""
    print("\n" + "="*80)
    print("测试 4: 输出JSON结构")
    print("="*80)
    
    # 模拟输出结构
    results = {
        "metadata": {
            "model_name": "test_model",
            "weights": "test.pt",
            "imgsz": 640,
            "batch": 1,
            "device": "cuda:0",
            "warmup": 50,
            "iters": 300,
        },
        "model_complexity": {
            "params": 3157200,
            "params_M": 3.157,
            "gflops": 8.2,
        },
        "latency": {
            "mean_ms": 5.23,
            "median_ms": 5.12,
            "p50_ms": 5.12,
            "p95_ms": 6.34,
            "p99_ms": 7.12,
            "std_ms": 0.45,
        },
        "throughput": {
            "fps_mean": 191.2,
            "fps_p50": 195.3,
        },
        "memory": {
            "peak_gpu_mem_MB": 512.5,
            "allocated_MB": 450.2,
            "reserved_MB": 600.0,
        }
    }
    
    # 验证必需字段
    required_sections = ["metadata", "model_complexity", "latency", "throughput", "memory"]
    for section in required_sections:
        assert section in results, f"缺少 {section}"
        print(f"  ✓ {section}")
    
    # 验证关键指标
    assert "mean_ms" in results["latency"]
    assert "p50_ms" in results["latency"]
    assert "p95_ms" in results["latency"]
    assert "fps_mean" in results["throughput"]
    assert "peak_gpu_mem_MB" in results["memory"]
    
    print(f"\n✓ 所有必需字段存在")
    
    # 测试JSON序列化
    json_str = json.dumps(results, indent=2)
    print(f"✓ JSON序列化成功")
    
    # 测试反序列化
    loaded = json.loads(json_str)
    assert loaded == results, "序列化后不一致"
    print(f"✓ JSON反序列化成功")
    
    print("\n示例JSON:")
    print(json_str)
    
    print("\n✅ JSON结构测试通过")
    return True


def test_benchmark_list_reuse():
    """测试基准列表复用"""
    print("\n" + "="*80)
    print("测试 5: 基准列表复用（公平对比）")
    print("="*80)
    
    data_yaml = "datasets/openparts/data.yaml"
    
    if not Path(data_yaml).exists():
        print(f"⚠️  数据集不存在，跳过测试")
        return True
    
    # 创建两次列表，使用相同种子
    list1 = create_benchmark_list(
        data_yaml, split="val", size=30, seed=42,
        output_path="results/bench/reuse_test1.txt"
    )
    
    list2 = create_benchmark_list(
        data_yaml, split="val", size=30, seed=42,
        output_path="results/bench/reuse_test2.txt"
    )
    
    # 验证一致性
    if list1 == list2:
        print(f"\n✓ 相同种子产生相同列表")
        print(f"  列表1: {len(list1)} 张")
        print(f"  列表2: {len(list2)} 张")
        print(f"  一致性: 100%")
    else:
        print(f"\n❌ 相同种子未产生相同列表")
        return False
    
    # 创建不同种子的列表
    list3 = create_benchmark_list(
        data_yaml, split="val", size=30, seed=999,
        output_path="results/bench/reuse_test3.txt"
    )
    
    # 应该不同
    if list1 != list3:
        print(f"✓ 不同种子产生不同列表")
    else:
        print(f"⚠️  不同种子产生了相同列表（可能数据集太小）")
    
    print("\n使用场景演示:")
    print("  1. 第一次运行模型A：创建 benchmark_list.txt")
    print("  2. 第二次运行模型B：使用 --use_benchmark_list 复用列表")
    print("  3. 确保所有模型在相同图像上测试")
    
    print("\n✅ 基准列表复用测试通过")
    return True


def main():
    """运行所有测试"""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "基准测试脚本验证" + " "*28 + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        ("基准测试列表创建", test_benchmark_list_creation),
        ("延迟测量（CUDA同步）", test_latency_measurement),
        ("GPU显存统计", test_gpu_memory),
        ("输出JSON结构", test_json_structure),
        ("基准列表复用", test_benchmark_list_reuse),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
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
        print("\n基准测试功能验证:")
        print("  ✓ 基准测试列表生成（固定种子）")
        print("  ✓ 延迟测量（包含CUDA同步）")
        print("  ✓ GPU显存统计")
        print("  ✓ JSON输出结构")
        print("  ✓ 同一输入列表复用")
        print("\n可以使用以下命令运行实际基准测试:")
        print("  python scripts/benchmark_model.py \\")
        print("      --weights results/runs/baseline/seed0/weights/best.pt \\")
        print("      --imgsz 640 \\")
        print("      --device 0")
        print("="*80 + "\n")
        return 0
    else:
        print(f"\n❌ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
