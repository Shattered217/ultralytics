"""
设置确定性种子 (Set Deterministic Seed)

该模块提供设置随机种子的函数，确保实验的可复现性。
必须在所有训练和评估脚本开始时调用 set_seed() 函数。

使用方法:
    from scripts.set_determinism import set_seed
    
    # 在训练/评估脚本开始时调用
    set_seed(seed=0)

关于 cuDNN 确定性的说明:
    - torch.backends.cudnn.deterministic = True: 强制使用确定性算法，保证可复现
    - torch.backends.cudnn.benchmark = False: 禁用自动优化，避免非确定性行为
    
    注意: 启用确定性可能会略微降低性能 (约 5-10%)，但对于科研实验，
          可复现性优先于性能。如果需要更快的速度，可以在确认实验设置后
          将 benchmark 设为 True，但需在报告中说明。
"""

import os
import random
import warnings

import numpy as np


def set_seed(seed: int = 0, deterministic: bool = True, benchmark: bool = False):
    """
    设置所有随机种子，确保实验可复现性
    
    Args:
        seed (int): 随机种子值，默认为 0。建议使用 [0, 1, 2] 进行多次实验
        deterministic (bool): 是否启用 PyTorch 确定性模式，默认为 True
            - True: 保证完全可复现，但可能略微降低性能
            - False: 允许非确定性算法，速度更快但结果可能略有差异
        benchmark (bool): 是否启用 cuDNN 自动优化，默认为 False
            - False: 禁用自动优化，保证可复现性
            - True: 启用自动优化，速度更快但可能导致结果不一致
    
    注意:
        - 对于科研实验，建议使用默认参数 (deterministic=True, benchmark=False)
        - 如果需要在多次运行中获得相同结果，必须设置相同的 seed
        - 多 GPU 训练时，还需要确保使用相同的进程数和分布式策略
    
    示例:
        >>> # 基本使用
        >>> set_seed(0)
        
        >>> # 在训练脚本中
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('--seed', type=int, default=0)
        >>> args = parser.parse_args()
        >>> set_seed(args.seed)
        
        >>> # 如果需要更快速度（牺牲部分可复现性）
        >>> set_seed(0, deterministic=False, benchmark=True)
    """
    # 设置 Python 内置 random 模块的种子
    random.seed(seed)
    
    # 设置 NumPy 的种子
    np.random.seed(seed)
    
    # 设置环境变量（某些库会读取）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 尝试设置 PyTorch 相关的种子
    try:
        import torch
        
        # 设置 PyTorch CPU 随机种子
        torch.manual_seed(seed)
        
        # 设置 PyTorch CUDA 随机种子（如果使用 GPU）
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # 对于多 GPU
            
            # 配置 cuDNN 行为
            # deterministic=True: 强制 cuDNN 使用确定性算法
            # 优点: 完全可复现
            # 缺点: 可能会略微降低性能（约 5-10%）
            torch.backends.cudnn.deterministic = deterministic
            
            # benchmark=False: 禁用 cuDNN 自动调优
            # 优点: 避免每次运行时选择不同的算法导致结果不一致
            # 缺点: 无法利用自动优化来提升速度
            torch.backends.cudnn.benchmark = benchmark
            
            if deterministic and benchmark:
                warnings.warn(
                    "同时设置 deterministic=True 和 benchmark=True 可能导致不确定性。"
                    "建议保持 benchmark=False 以确保可复现性。",
                    UserWarning
                )
        
        print(f"✅ 已设置随机种子: {seed}")
        print(f"   - Python random: ✓")
        print(f"   - NumPy: ✓")
        print(f"   - PyTorch CPU: ✓")
        if torch.cuda.is_available():
            print(f"   - PyTorch CUDA: ✓")
            print(f"   - cuDNN deterministic: {deterministic}")
            print(f"   - cuDNN benchmark: {benchmark}")
        
    except ImportError:
        warnings.warn(
            "PyTorch 未安装，仅设置了 Python 和 NumPy 的随机种子。",
            UserWarning
        )
        print(f"⚠️  已设置随机种子: {seed} (仅 Python 和 NumPy)")


def check_determinism():
    """
    检查当前环境的确定性设置
    
    该函数用于调试，检查所有随机数生成器的状态和 PyTorch 的确定性配置。
    
    示例:
        >>> set_seed(42)
        >>> check_determinism()
    """
    print("\n" + "=" * 60)
    print("确定性设置检查 (Determinism Check)")
    print("=" * 60)
    
    # 检查环境变量
    pythonhashseed = os.environ.get('PYTHONHASHSEED', 'not set')
    print(f"PYTHONHASHSEED: {pythonhashseed}")
    
    try:
        import torch
        print(f"\nPyTorch 版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"CUDA 可用: ✓")
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
            print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
            print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"CUDA 可用: ✗ (CPU only)")
        
        # 测试随机数生成
        print(f"\n随机数生成测试:")
        print(f"  Python random: {random.random():.6f}")
        print(f"  NumPy random: {np.random.rand():.6f}")
        print(f"  PyTorch CPU: {torch.rand(1).item():.6f}")
        if torch.cuda.is_available():
            print(f"  PyTorch CUDA: {torch.rand(1, device='cuda').item():.6f}")
        
    except ImportError:
        print("\n⚠️  PyTorch 未安装，无法检查 CUDA/cuDNN 设置")
    
    print("=" * 60 + "\n")


def seed_worker(worker_id):
    """
    DataLoader worker 初始化函数，确保每个 worker 的随机性可控
    
    在使用 PyTorch DataLoader 时，如果设置了 num_workers > 0，
    每个 worker 进程都有独立的随机状态。使用此函数可以确保：
    1. 每个 worker 的随机种子是确定的
    2. 不同 worker 之间的随机种子不同（避免重复）
    3. 多次运行时，相同 worker_id 获得相同的数据增强
    
    Args:
        worker_id (int): DataLoader worker 的 ID
    
    使用方法:
        >>> from torch.utils.data import DataLoader
        >>> from scripts.set_determinism import set_seed, seed_worker
        >>> 
        >>> set_seed(0)  # 主进程种子
        >>> 
        >>> # 创建 DataLoader 时传入 worker_init_fn
        >>> dataloader = DataLoader(
        ...     dataset,
        ...     batch_size=16,
        ...     num_workers=4,
        ...     worker_init_fn=seed_worker  # 关键：确保 worker 可复现
        ... )
    
    注意:
        - 这个函数会在每个 worker 进程启动时自动调用
        - worker_id 从 0 开始，范围是 [0, num_workers-1]
        - 每个 worker 的种子 = 主进程种子 + worker_id（需要主进程先调用 set_seed）
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def test_reproducibility(seed: int = 42, n_runs: int = 3):
    """
    测试随机种子设置是否有效（用于调试）
    
    该函数会多次设置相同的种子并生成随机数，验证结果是否一致。
    
    Args:
        seed (int): 测试用的随机种子
        n_runs (int): 测试运行次数
    
    示例:
        >>> test_reproducibility(seed=42, n_runs=3)
    """
    print(f"\n🧪 测试可复现性 (种子={seed}, 运行{n_runs}次)...")
    print("-" * 60)
    
    results = []
    
    for run in range(n_runs):
        # 每次都重新设置相同的种子
        set_seed(seed, deterministic=True, benchmark=False)
        
        # 生成一些随机数
        python_rand = random.random()
        numpy_rand = np.random.rand()
        
        try:
            import torch
            torch_cpu_rand = torch.rand(1).item()
            if torch.cuda.is_available():
                torch_cuda_rand = torch.rand(1, device='cuda').item()
            else:
                torch_cuda_rand = None
        except ImportError:
            torch_cpu_rand = None
            torch_cuda_rand = None
        
        results.append({
            'python': python_rand,
            'numpy': numpy_rand,
            'torch_cpu': torch_cpu_rand,
            'torch_cuda': torch_cuda_rand
        })
        
        print(f"运行 {run + 1}:")
        print(f"  Python: {python_rand:.10f}")
        print(f"  NumPy:  {numpy_rand:.10f}")
        if torch_cpu_rand is not None:
            print(f"  PyTorch CPU: {torch_cpu_rand:.10f}")
        if torch_cuda_rand is not None:
            print(f"  PyTorch CUDA: {torch_cuda_rand:.10f}")
    
    # 检查所有运行的结果是否一致
    print("\n" + "-" * 60)
    all_identical = True
    for key in ['python', 'numpy', 'torch_cpu', 'torch_cuda']:
        if results[0][key] is None:
            continue
        values = [r[key] for r in results]
        is_identical = all(v == values[0] for v in values)
        status = "✅ 一致" if is_identical else "❌ 不一致"
        print(f"{key:15s}: {status}")
        if not is_identical:
            all_identical = False
    
    print("-" * 60)
    if all_identical:
        print("✅ 可复现性测试通过！所有运行结果完全一致。")
    else:
        print("❌ 可复现性测试失败！不同运行的结果不一致。")
        print("   请检查 deterministic 和 benchmark 设置。")
    print()


if __name__ == "__main__":
    """
    直接运行此脚本可以测试种子设置功能
    """
    print("=" * 60)
    print("随机种子设置工具 (Set Determinism Tool)")
    print("=" * 60)
    
    # 测试基本功能
    print("\n1️⃣ 设置种子...")
    set_seed(seed=42, deterministic=True, benchmark=False)
    
    # 检查确定性配置
    print("\n2️⃣ 检查确定性配置...")
    check_determinism()
    
    # 测试可复现性
    print("\n3️⃣ 测试可复现性...")
    test_reproducibility(seed=42, n_runs=3)
    
    print("✨ 测试完成！")
    print("\n使用提示:")
    print("  在训练脚本中添加:")
    print("    from scripts.set_determinism import set_seed")
    print("    set_seed(args.seed)")
