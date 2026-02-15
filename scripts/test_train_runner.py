"""
训练运行器验证脚本 (Dry-run Test)

测试训练运行器的参数解析、配置加载和快照创建功能，不实际运行训练。

用法：
    python scripts/test_train_runner.py
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import shutil
from datetime import datetime
from scripts.set_determinism import set_seed
from scripts.load_config import load_config, save_config
from scripts.record_env import record_environment


def get_relative_path(path):
    """安全地获取相对路径"""
    try:
        return Path(path).relative_to(Path.cwd())
    except ValueError:
        return Path(path)


def test_dry_run():
    """测试训练运行器的前置步骤（不实际训练）"""
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*22 + "训练运行器验证测试" + " "*32 + "║")
    print("╚" + "="*78 + "╝\n")
    
    # 测试参数
    exp_name = "test_baseline"
    model_yaml = "ultralytics/cfg/models/v8/yolov8n_baseline.yaml"
    seed = 999
    train_cfg = "experiments/base_train.yaml"
    
    print(f"{'='*80}")
    print("测试配置")
    print(f"{'='*80}")
    print(f"实验名称: {exp_name}")
    print(f"模型配置: {model_yaml}")
    print(f"随机种子: {seed}")
    print(f"训练配置: {train_cfg}")
    print()
    
    # 步骤 1: 设置随机种子
    print(f"{'='*80}")
    print("步骤 1: 测试随机种子设置")
    print(f"{'='*80}")
    try:
        set_seed(seed)
        print(f"✓ 随机种子设置成功: {seed}\n")
    except Exception as e:
        print(f"❌ 随机种子设置失败: {e}\n")
        return False
    
    # 步骤 2: 创建快照目录
    print(f"{'='*80}")
    print("步骤 2: 测试快照目录创建")
    print(f"{'='*80}")
    snapshot_dir = Path("results/runs") / exp_name / f"seed{seed}"
    
    # 清理旧测试目录
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
        print(f"✓ 已清理旧测试目录")
    
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ 快照目录已创建: {get_relative_path(snapshot_dir)}\n")
    
    # 步骤 3: 加载训练配置
    print(f"{'='*80}")
    print("步骤 3: 测试训练配置加载")
    print(f"{'='*80}")
    try:
        train_config = load_config(train_cfg, validate=True)
        print(f"✓ 基础配置已加载: {train_cfg}")
        print(f"  - 轮数: {train_config.get('epochs')}")
        print(f"  - 批次: {train_config.get('batch')}")
        print(f"  - 学习率: {train_config.get('lr0')}")
        
        # 应用测试覆盖
        train_config['model'] = model_yaml
        train_config['name'] = f"{exp_name}_seed{seed}"
        train_config['data'] = "datasets/openparts/data.yaml"  # 测试覆盖
        train_config['epochs'] = 2  # 测试短训练
        print(f"✓ 配置参数已覆盖（测试）\n")
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}\n")
        return False
    
    # 步骤 4: 创建实验快照
    print(f"{'='*80}")
    print("步骤 4: 测试实验快照创建")
    print(f"{'='*80}")
    
    try:
        # 4.1 保存模型配置副本
        model_yaml_path = Path(model_yaml)
        if not model_yaml_path.exists():
            print(f"❌ 模型配置文件不存在: {model_yaml}")
            return False
        
        model_copy = snapshot_dir / "model_config.yaml"
        shutil.copy2(model_yaml_path, model_copy)
        print(f"✓ 模型配置已保存: {get_relative_path(model_copy)}")
        
        # 4.2 保存训练配置
        train_config_resolved = snapshot_dir / "resolved_train.yaml"
        save_config(train_config, train_config_resolved)
        print(f"✓ 训练配置已保存: {get_relative_path(train_config_resolved)}")
        
        # 4.3 保存环境信息
        env_json_path = "results/metadata/env.json"
        env_copy = snapshot_dir / "env.json"
        
        if Path(env_json_path).exists():
            shutil.copy2(env_json_path, env_copy)
            print(f"✓ 环境信息已复制: {get_relative_path(env_copy)}")
        else:
            print(f"⚠️  环境信息不存在，将重新记录")
            record_environment(str(env_copy))
            print(f"✓ 环境信息已记录: {get_relative_path(env_copy)}")
        
        # 4.4 保存元数据
        metadata = {
            "exp_name": exp_name,
            "seed": seed,
            "model_yaml": str(model_yaml),
            "snapshot_time": datetime.now().isoformat(),
            "snapshot_dir": str(get_relative_path(snapshot_dir)),
            "test_mode": True,
        }
        metadata_path = snapshot_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✓ 元数据已保存: {get_relative_path(metadata_path)}\n")
        
    except Exception as e:
        print(f"❌ 快照创建失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    
    # 步骤 5: 验证快照完整性
    print(f"{'='*80}")
    print("步骤 5: 验证快照完整性")
    print(f"{'='*80}")
    
    required_files = [
        "model_config.yaml",
        "resolved_train.yaml",
        "env.json",
        "metadata.json",
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = snapshot_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"✓ {filename} 存在 ({size_kb:.2f} KB)")
        else:
            print(f"❌ {filename} 不存在")
            all_exist = False
    
    print()
    
    if not all_exist:
        print("❌ 快照文件不完整")
        return False
    
    # 步骤 6: 创建模拟的 metrics_train.json
    print(f"{'='*80}")
    print("步骤 6: 测试指标保存")
    print(f"{'='*80}")
    
    try:
        mock_metrics = {
            "extraction_time": datetime.now().isoformat(),
            "test_mode": True,
            "from_csv": {
                "mAP50": 0.6234,
                "mAP50-95": 0.4567,
                "precision": 0.7123,
                "recall": 0.5894,
                "box_loss": 1.234,
                "cls_loss": 0.567,
                "dfl_loss": 0.890,
            }
        }
        
        metrics_file = snapshot_dir / "metrics_train.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(mock_metrics, f, indent=2, ensure_ascii=False)
        print(f"✓ 模拟指标已保存: {get_relative_path(metrics_file)}")
        
        # 打印模拟指标
        print(f"\n模拟指标:")
        for key, value in mock_metrics["from_csv"].items():
            print(f"  {key}: {value}")
        print()
        
    except Exception as e:
        print(f"❌ 指标保存失败: {e}\n")
        return False
    
    # 最终总结
    print(f"{'='*80}")
    print("验证测试总结")
    print(f"{'='*80}")
    print(f"✅ 快照目录: {get_relative_path(snapshot_dir)}")
    print(f"✅ 所有前置步骤测试通过")
    print(f"\n快照内容:")
    for filename in required_files:
        print(f"  ✓ {filename}")
    print(f"  ✓ metrics_train.json (模拟)")
    print(f"\n注意: 这是 dry-run 测试，未实际训练模型")
    print(f"{'='*80}\n")
    
    # 询问是否清理测试目录
    try:
        response = input("是否清理测试快照目录？(y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        response = 'n'
        print()
    
    if response == 'y':
        shutil.rmtree(snapshot_dir.parent.parent)  # 删除整个 test_baseline 目录
        print(f"✓ 已清理测试目录\n")
    else:
        print(f"✓ 测试快照保留在: {snapshot_dir}\n")
    
    return True


def main():
    """主函数"""
    print("\n测试训练运行器的配置加载和快照创建功能\n")
    
    success = test_dry_run()
    
    if success:
        print("🎉 训练运行器验证测试通过！")
        print("\n可以使用以下命令运行实际训练:")
        print("  python scripts/run_train_one.py \\")
        print("      --exp_name baseline \\")
        print("      --model_yaml ultralytics/cfg/models/v8/yolov8n_baseline.yaml \\")
        print("      --seed 0")
        return 0
    else:
        print("❌ 训练运行器验证测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
