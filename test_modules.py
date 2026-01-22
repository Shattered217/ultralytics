"""
测试脚本：验证所有自定义模块是否正常工作
"""

import sys
import os

# 添加本地 ultralytics 到路径
sys.path.insert(0, os.path.abspath("../ultralytics"))

import torch
from ultralytics.nn.modules import (
    GSConv, 
    VoVGSCSP, 
    DySample_Simple,
    SPDConv,
    EMA
)


def test_module(module_name, module, input_tensor, expected_output_shape=None):
    """测试单个模块"""
    print(f"\n{'='*60}")
    print(f"测试模块: {module_name}")
    print(f"{'='*60}")
    
    try:
        # 前向传播
        with torch.no_grad():
            output = module(input_tensor)
        
        # 检查输出
        print(f"✅ 前向传播成功")
        print(f"   输入形状:  {list(input_tensor.shape)}")
        print(f"   输出形状:  {list(output.shape)}")
        
        if expected_output_shape:
            assert output.shape == expected_output_shape, \
                f"输出形状不匹配: 期望 {expected_output_shape}, 实际 {output.shape}"
            print(f"   形状验证:  ✅ 通过")
        
        # 计算参数量
        params = sum(p.numel() for p in module.parameters())
        print(f"   参数量:    {params:,} ({params/1e6:.2f}M)")
        
        # 计算 FLOPs（简单估计）
        if hasattr(module, 'flops'):
            flops = module.flops(input_tensor)
            print(f"   FLOPs:     {flops:,} ({flops/1e9:.2f}G)")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("GSE-YOLOv8 模块测试")
    print("=" * 60)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")  # 强制改为 CPU 运行
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    results = {}
    
    # ========== 测试 1: GSConv ==========
    print("\n\n" + "🔧 轻量化卷积模块")
    x = torch.randn(2, 128, 40, 40).to(device)
    
    gsconv = GSConv(128, 256, k=3, s=1).to(device)
    results['GSConv'] = test_module(
        'GSConv',
        gsconv,
        x,
        expected_output_shape=torch.Size([2, 256, 40, 40])
    )
    
    # ========== 测试 2: VoVGSCSP ==========
    print("\n\n" + "🔧 轻量化 CSP 模块")
    x = torch.randn(2, 128, 40, 40).to(device)
    
    vov_csp = VoVGSCSP(128, 256, n=3).to(device)
    results['VoVGSCSP'] = test_module(
        'VoVGSCSP',
        vov_csp,
        x,
        expected_output_shape=torch.Size([2, 256, 40, 40])
    )
    
    # ========== 测试 3: DySample_Simple ==========
    print("\n\n" + "🔧 动态上采样模块")
    x = torch.randn(2, 256, 20, 20).to(device)
    
    dysample = DySample_Simple(256, scale=2, groups=4).to(device)
    results['DySample_Simple'] = test_module(
        'DySample_Simple',
        dysample,
        x,
        expected_output_shape=torch.Size([2, 256, 40, 40])
    )
    
    # ========== 测试 4: SPDConv ==========
    print("\n\n" + "🔧 无损下采样模块 (SPD-Conv)")
    x = torch.randn(2, 64, 80, 80).to(device)
    
    spdconv = SPDConv(64, 128).to(device)
    results['SPDConv'] = test_module(
        'SPDConv',
        spdconv,
        x,
        expected_output_shape=torch.Size([2, 128, 40, 40])
    )
    
    # ========== 测试 5: EMA ==========
    print("\n\n" + "🔧 高效注意力模块 (EMA)")
    x = torch.randn(2, 256, 40, 40).to(device)
    
    ema = EMA(256).to(device)
    results['EMA'] = test_module(
        'EMA',
        ema,
        x,
        expected_output_shape=torch.Size([2, 256, 40, 40])
    )
    
    # ========== 测试 6: 组合测试 ==========
    print("\n\n" + "🔧 组合模块测试")
    print("=" * 60)
    
    try:
        x = torch.randn(1, 128, 80, 80).to(device)
        
        # 模拟 Neck 中的特征融合流程
        print("\n模拟 Neck 流程:")
        print("  输入: [1, 128, 80, 80]")
        
        # 1. 上采样
        up = DySample_Simple(128, scale=2).to(device)
        x1 = up(x)
        print(f"  ↓ DySample(2x): {list(x1.shape)}")
        
        # 2. 假设 concat 另一个分支
        x2 = torch.randn(1, 128, 160, 160).to(device)
        x_cat = torch.cat([x1, x2], dim=1)
        print(f"  ↓ Concat:       {list(x_cat.shape)}")
        
        # 3. VoVGSCSP 融合
        fusion = VoVGSCSP(256, 128, n=2).to(device)
        x3 = fusion(x_cat)
        print(f"  ↓ VoVGSCSP:     {list(x3.shape)}")
        
        # 4. EMA 增强
        ema = EMA(128).to(device)
        x4 = ema(x3)
        print(f"  ↓ EMA:          {list(x4.shape)}")
        
        print("\n✅ 组合测试通过")
        results['Combination'] = True
        
    except Exception as e:
        print(f"\n❌ 组合测试失败: {str(e)}")
        results['Combination'] = False
    
    # ========== 汇总结果 ==========
    print("\n\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name:<20} {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！可以开始训练了。")
    else:
        print("⚠️  部分测试失败，请检查模块实现。")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
