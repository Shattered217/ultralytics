"""
模块单元测试 (Module Unit Tests)

测试自定义模块的前向传播，确保：
1. 输出 shape 与预期一致
2. 无运行时异常
3. 梯度可以正常反向传播

用法：
    python scripts/test_modules.py
    python scripts/test_modules.py --verbose  # 显示详细信息
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn


def test_ghostconv():
    """测试 GhostConv 模块"""
    from ultralytics.nn.modules import GhostConv
    
    print("\n" + "="*80)
    print("测试 GhostConv 模块")
    print("="*80)
    
    # 测试参数
    batch_size = 2
    c_in = 64
    c_out = 128
    h, w = 32, 32
    
    # 创建模块
    module = GhostConv(c_in, c_out, k=3, s=1)
    print(f"✓ 模块创建成功: GhostConv({c_in}, {c_out}, k=3, s=1)")
    
    # 创建输入
    x = torch.randn(batch_size, c_in, h, w)
    print(f"✓ 输入 shape: {x.shape}")
    
    # 前向传播
    try:
        y = module(x)
        print(f"✓ 输出 shape: {y.shape}")
        
        # 检查输出 shape
        expected_shape = (batch_size, c_out, h, w)
        assert y.shape == expected_shape, f"输出 shape 不匹配！预期 {expected_shape}，实际 {y.shape}"
        print(f"✓ Shape 验证通过")
        
        # 测试梯度反向传播
        loss = y.sum()
        loss.backward()
        print(f"✓ 梯度反向传播成功")
        
        print("✅ GhostConv 测试通过\n")
        return True
        
    except Exception as e:
        print(f"❌ GhostConv 测试失败: {e}\n")
        return False


def test_ghostbottleneck():
    """测试 GhostBottleneck 模块"""
    from ultralytics.nn.modules import GhostBottleneck
    
    print("\n" + "="*80)
    print("测试 GhostBottleneck 模块")
    print("="*80)
    
    # 测试参数
    batch_size = 2
    c_in = 128
    c_out = 128
    h, w = 32, 32
    
    # 测试 stride=1 (with shortcut)
    module = GhostBottleneck(c_in, c_out, k=3, s=1)
    print(f"✓ 模块创建成功: GhostBottleneck({c_in}, {c_out}, k=3, s=1)")
    
    x = torch.randn(batch_size, c_in, h, w)
    print(f"✓ 输入 shape: {x.shape}")
    
    try:
        y = module(x)
        print(f"✓ 输出 shape: {y.shape}")
        
        expected_shape = (batch_size, c_out, h, w)
        assert y.shape == expected_shape, f"输出 shape 不匹配！预期 {expected_shape}，实际 {y.shape}"
        print(f"✓ Shape 验证通过 (s=1)")
        
        # 测试 stride=2 (with downsampling)
        module_s2 = GhostBottleneck(c_in, c_out, k=3, s=2)
        y_s2 = module_s2(x)
        expected_shape_s2 = (batch_size, c_out, h // 2, w // 2)
        assert y_s2.shape == expected_shape_s2, f"输出 shape 不匹配！预期 {expected_shape_s2}，实际 {y_s2.shape}"
        print(f"✓ Shape 验证通过 (s=2)")
        
        # 测试梯度反向传播
        loss = y.sum() + y_s2.sum()
        loss.backward()
        print(f"✓ 梯度反向传播成功")
        
        print("✅ GhostBottleneck 测试通过\n")
        return True
        
    except Exception as e:
        print(f"❌ GhostBottleneck 测试失败: {e}\n")
        return False


def test_c3ghost():
    """测试 C3Ghost 模块"""
    from ultralytics.nn.modules import C3Ghost
    
    print("\n" + "="*80)
    print("测试 C3Ghost 模块")
    print("="*80)
    
    # 测试参数
    batch_size = 2
    c_in = 256
    c_out = 256
    h, w = 16, 16
    n = 3  # number of bottlenecks
    
    # 创建模块
    module = C3Ghost(c_in, c_out, n=n, shortcut=True)
    print(f"✓ 模块创建成功: C3Ghost({c_in}, {c_out}, n={n})")
    
    # 创建输入
    x = torch.randn(batch_size, c_in, h, w)
    print(f"✓ 输入 shape: {x.shape}")
    
    # 前向传播
    try:
        y = module(x)
        print(f"✓ 输出 shape: {y.shape}")
        
        # 检查输出 shape
        expected_shape = (batch_size, c_out, h, w)
        assert y.shape == expected_shape, f"输出 shape 不匹配！预期 {expected_shape}，实际 {y.shape}"
        print(f"✓ Shape 验证通过")
        
        # 测试梯度反向传播
        loss = y.sum()
        loss.backward()
        print(f"✓ 梯度反向传播成功")
        
        print("✅ C3Ghost 测试通过\n")
        return True
        
    except Exception as e:
        print(f"❌ C3Ghost 测试失败: {e}\n")
        return False


def test_eca():
    """测试 ECA 模块"""
    from ultralytics.nn.modules import ECA
    
    print("\n" + "="*80)
    print("测试 ECA (Efficient Channel Attention) 模块")
    print("="*80)
    
    # 测试参数
    batch_size = 2
    channels = 256
    h, w = 32, 32
    
    # 测试不同的 kernel size
    for k in [3, 5, 7]:
        print(f"\n--- 测试 kernel_size={k} ---")
        
        # 创建模块
        module = ECA(channels, kernel_size=k)
        print(f"✓ 模块创建成功: ECA({channels}, kernel_size={k})")
        
        # 创建输入
        x = torch.randn(batch_size, channels, h, w)
        print(f"✓ 输入 shape: {x.shape}")
        
        # 前向传播
        try:
            y = module(x)
            print(f"✓ 输出 shape: {y.shape}")
            
            # 检查输出 shape（ECA 保持输入 shape 不变）
            assert y.shape == x.shape, f"输出 shape 不匹配！预期 {x.shape}，实际 {y.shape}"
            print(f"✓ Shape 验证通过")
            
            # 检查注意力机制是否生效（输出不应该与输入完全相同）
            assert not torch.allclose(x, y), "输出与输入完全相同，注意力机制可能未生效"
            print(f"✓ 注意力加权生效")
            
            # 测试梯度反向传播
            loss = y.sum()
            loss.backward()
            print(f"✓ 梯度反向传播成功")
            
        except Exception as e:
            print(f"❌ ECA (k={k}) 测试失败: {e}\n")
            return False
    
    print("\n✅ ECA 测试通过\n")
    return True


def test_eca_with_conv():
    """测试 ECA 与卷积层的组合"""
    from ultralytics.nn.modules import ECA, Conv
    
    print("\n" + "="*80)
    print("测试 Conv + ECA 组合")
    print("="*80)
    
    # 测试参数
    batch_size = 2
    c_in = 128
    c_out = 256
    h, w = 32, 32
    
    # 创建组合模块
    class ConvECA(nn.Module):
        def __init__(self, c1, c2):
            super().__init__()
            self.conv = Conv(c1, c2, k=3, s=1)
            self.eca = ECA(c2, kernel_size=3)
        
        def forward(self, x):
            return self.eca(self.conv(x))
    
    module = ConvECA(c_in, c_out)
    print(f"✓ 模块创建成功: Conv({c_in}, {c_out}) + ECA({c_out})")
    
    # 创建输入
    x = torch.randn(batch_size, c_in, h, w)
    print(f"✓ 输入 shape: {x.shape}")
    
    # 前向传播
    try:
        y = module(x)
        print(f"✓ 输出 shape: {y.shape}")
        
        # 检查输出 shape
        expected_shape = (batch_size, c_out, h, w)
        assert y.shape == expected_shape, f"输出 shape 不匹配！预期 {expected_shape}，实际 {y.shape}"
        print(f"✓ Shape 验证通过")
        
        # 测试梯度反向传播
        loss = y.sum()
        loss.backward()
        print(f"✓ 梯度反向传播成功")
        
        print("✅ Conv + ECA 组合测试通过\n")
        return True
        
    except Exception as e:
        print(f"❌ Conv + ECA 组合测试失败: {e}\n")
        return False


def print_module_info():
    """打印模块信息"""
    from ultralytics.nn.modules import GhostConv, GhostBottleneck, C3Ghost, ECA
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "模块信息汇总" + " "*41 + "║")
    print("╚" + "="*78 + "╝\n")
    
    modules = [
        ("GhostConv", GhostConv, "Ghost 卷积，生成更多特征但参数更少"),
        ("GhostBottleneck", GhostBottleneck, "Ghost 瓶颈块，轻量级残差结构"),
        ("C3Ghost", C3Ghost, "C3 模块的 Ghost 版本"),
        ("ECA", ECA, "高效通道注意力，使用 1D 卷积实现跨通道交互"),
    ]
    
    for name, cls, desc in modules:
        print(f"📦 {name}")
        print(f"   描述: {desc}")
        print(f"   来源: {cls.__module__}.{cls.__name__}")
        print()


def main():
    """运行所有测试"""
    parser = argparse.ArgumentParser(description="测试自定义模块")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    args = parser.parse_args()
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "Ultralytics 自定义模块单元测试" + " "*25 + "║")
    print("╚" + "="*78 + "╝")
    
    if args.verbose:
        print_module_info()
    
    # 运行所有测试
    tests = [
        ("GhostConv", test_ghostconv),
        ("GhostBottleneck", test_ghostbottleneck),
        ("C3Ghost", test_c3ghost),
        ("ECA", test_eca),
        ("Conv + ECA 组合", test_eca_with_conv),
    ]
    
    results = []
    for name, test_func in tests:
        passed = test_func()
        results.append((name, passed))
    
    # 输出测试结果汇总
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*30 + "测试结果汇总" + " "*34 + "║")
    print("╚" + "="*78 + "╝\n")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status}: {name}")
    
    print(f"\n总计: {passed_count}/{total_count} 个测试通过")
    
    if passed_count == total_count:
        print("\n🎉 所有测试通过！模块可以正常使用。\n")
        return 0
    else:
        print(f"\n⚠️  有 {total_count - passed_count} 个测试失败，请检查模块实现。\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
