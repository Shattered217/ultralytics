"""
验证模型配置文件 (Validate Model Configurations)

测试所有消融实验的模型 YAML 配置文件能否被 YOLO 正常加载和构建。

用法：
    python scripts/validate_models.py
    python scripts/validate_models.py --verbose    # 显示详细模型信息
    python scripts/validate_models.py --model yolov8n_baseline.yaml  # 测试单个模型
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from ultralytics import YOLO


def validate_model(model_path, verbose=False):
    """验证单个模型配置"""
    model_name = Path(model_path).stem
    
    print(f"\n{'='*80}")
    print(f"测试模型: {model_name}")
    print(f"配置文件: {model_path}")
    print(f"{'='*80}")
    
    try:
        # 加载模型
        model = YOLO(model_path)
        print(f"✓ 模型加载成功")
        
        # 打印模型信息
        if verbose:
            print(f"\n--- 模型结构信息 ---")
            print(model.model)
            print(f"\n--- 模型参数统计 ---")
            
        # 获取模型参数量
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        print(f"\n模型统计:")
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
        print(f"  - 模型层数: {len(list(model.model.modules()))}")
        
        # 测试前向传播
        import torch
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model.model(dummy_input)
        print(f"✓ 前向传播测试通过")
        print(f"  - 输入 shape: {dummy_input.shape}")
        if isinstance(output, (list, tuple)):
            print(f"  - 输出数量: {len(output)}")
            for i, out in enumerate(output):
                if hasattr(out, 'shape'):
                    print(f"    输出 {i} shape: {out.shape}")
        else:
            print(f"  - 输出 shape: {output.shape}")
        
        print(f"\n✅ {model_name} 验证通过\n")
        return True, total_params, trainable_params
        
    except Exception as e:
        print(f"\n❌ {model_name} 验证失败:")
        print(f"错误信息: {e}")
        import traceback
        if verbose:
            print("\n详细错误信息:")
            traceback.print_exc()
        print()
        return False, 0, 0


def main():
    """验证所有模型配置"""
    parser = argparse.ArgumentParser(description="验证模型配置文件")
    parser.add_argument("--model", type=str, help="指定单个模型配置文件")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    args = parser.parse_args()
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*22 + "模型配置文件验证" + " "*32 + "║")
    print("╚" + "="*78 + "╝")
    
    # 定义所有消融实验模型
    models = [
        ("E0: Baseline", "ultralytics/cfg/models/v8/yolov8n_baseline.yaml"),
        ("E1: Ghost", "ultralytics/cfg/models/v8/yolov8n_ghost.yaml"),
        ("E2: ECA", "ultralytics/cfg/models/v8/yolov8n_eca.yaml"),
        ("E3: P2 Head", "ultralytics/cfg/models/v8/yolov8n_p2.yaml"),
        ("E4: Ghost+ECA", "ultralytics/cfg/models/v8/yolov8n_ghost_eca.yaml"),
        ("E7: Full (Ghost+ECA+P2)", "ultralytics/cfg/models/v8/yolov8n_ghost_eca_p2.yaml"),
    ]
    
    # 如果指定了单个模型
    if args.model:
        model_path = args.model
        if not Path(model_path).exists():
            # 尝试在 cfg/models/v8 目录下查找
            model_path = f"ultralytics/cfg/models/v8/{args.model}"
        
        if not Path(model_path).exists():
            print(f"❌ 找不到模型配置文件: {args.model}")
            return 1
        
        success, _, _ = validate_model(model_path, args.verbose)
        return 0 if success else 1
    
    # 验证所有模型
    results = []
    for exp_name, model_path in models:
        if not Path(model_path).exists():
            print(f"\n⚠️  跳过 {exp_name}: 配置文件不存在 ({model_path})")
            results.append((exp_name, False, 0, 0))
            continue
        
        success, total_params, trainable_params = validate_model(model_path, args.verbose)
        results.append((exp_name, success, total_params, trainable_params))
    
    # 输出验证结果汇总
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*28 + "验证结果汇总" + " "*36 + "║")
    print("╚" + "="*78 + "╝\n")
    
    print(f"{'实验名称':<30} {'状态':<10} {'总参数量':>15} {'可训练参数':>15}")
    print("-" * 80)
    
    passed_count = 0
    for exp_name, success, total_params, trainable_params in results:
        status = "✅ 通过" if success else "❌ 失败"
        params_str = f"{total_params:,}" if success else "N/A"
        train_str = f"{trainable_params:,}" if success else "N/A"
        print(f"{exp_name:<30} {status:<10} {params_str:>15} {train_str:>15}")
        if success:
            passed_count += 1
    
    print("-" * 80)
    print(f"\n总计: {passed_count}/{len(results)} 个模型验证通过")
    
    # 参数量对比分析
    if passed_count > 0:
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*28 + "参数量对比" + " "*38 + "║")
        print("╚" + "="*78 + "╝\n")
        
        baseline_params = None
        for exp_name, success, total_params, trainable_params in results:
            if success:
                if "Baseline" in exp_name:
                    baseline_params = total_params
                    print(f"{exp_name}: {total_params:,} (基准)")
                elif baseline_params:
                    diff = total_params - baseline_params
                    ratio = (total_params / baseline_params - 1) * 100
                    sign = "+" if diff > 0 else ""
                    print(f"{exp_name}: {total_params:,} ({sign}{diff:,}, {sign}{ratio:.2f}%)")
                else:
                    print(f"{exp_name}: {total_params:,}")
    
    if passed_count == len(results):
        print("\n🎉 所有模型配置验证通过！可以开始训练实验。\n")
        return 0
    else:
        print(f"\n⚠️  有 {len(results) - passed_count} 个模型验证失败，请检查配置文件。\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
