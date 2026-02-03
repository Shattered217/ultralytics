"""
严格可追踪的训练运行器 (Traceable Training Runner)

为单个实验运行一次训练，记录完整的实验快照，确保可复现性。

用法：
    python scripts/run_train_one.py --exp_name baseline --model_yaml ultralytics/cfg/models/v8/yolov8n_baseline.yaml --seed 0
    python scripts/run_train_one.py --exp_name ghost --model_yaml ultralytics/cfg/models/v8/yolov8n_ghost.yaml --seed 0 --data datasets/selfparts/data.yaml
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import shutil
from datetime import datetime
from ultralytics import YOLO

from scripts.set_determinism import set_seed
from scripts.load_config import load_config, save_config
from scripts.record_env import record_environment


def create_experiment_snapshot(snapshot_dir, model_yaml, train_config, env_json_path):
    """创建实验快照，保存所有配置文件"""
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("创建实验快照")
    print(f"{'='*80}")
    
    # 1. 保存模型配置副本
    model_yaml_path = Path(model_yaml)
    model_copy = snapshot_dir / "model_config.yaml"
    shutil.copy2(model_yaml_path, model_copy)
    try:
        rel_path = model_copy.relative_to(Path.cwd())
    except ValueError:
        rel_path = model_copy
    print(f"✓ 模型配置已保存: {rel_path}")
    
    # 2. 保存实际生效的训练配置
    train_config_resolved = snapshot_dir / "resolved_train.yaml"
    save_config(train_config, train_config_resolved)
    try:
        rel_path = train_config_resolved.relative_to(Path.cwd())
    except ValueError:
        rel_path = train_config_resolved
    print(f"✓ 训练配置已保存: {rel_path}")
    
    # 3. 保存或引用环境信息
    if env_json_path and Path(env_json_path).exists():
        env_copy = snapshot_dir / "env.json"
        shutil.copy2(env_json_path, env_copy)
        try:
            rel_path = env_copy.relative_to(Path.cwd())
        except ValueError:
            rel_path = env_copy
        print(f"✓ 环境信息已保存: {rel_path}")
    else:
        print(f"⚠️  环境信息文件不存在，将重新记录")
        env_copy = snapshot_dir / "env.json"
        record_environment(str(env_copy))
        try:
            rel_path = env_copy.relative_to(Path.cwd())
        except ValueError:
            rel_path = env_copy
        print(f"✓ 环境信息已重新记录: {rel_path}")
    
    # 4. 保存实验元数据
    metadata = {
        "exp_name": snapshot_dir.parent.parent.name,
        "seed": int(snapshot_dir.name.replace("seed", "")),
        "model_yaml": str(model_yaml),
        "snapshot_time": datetime.now().isoformat(),
        "snapshot_dir": str(snapshot_dir),
    }
    metadata_path = snapshot_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    try:
        rel_path = metadata_path.relative_to(Path.cwd())
    except ValueError:
        rel_path = metadata_path
    print(f"✓ 元数据已保存: {rel_path}")
    
    print(f"{'='*80}\n")
    
    return {
        "model_config": model_copy,
        "train_config": train_config_resolved,
        "env": env_copy,
        "metadata": metadata_path,
    }


def extract_metrics(results, results_dir):
    """从训练结果中提取关键指标"""
    metrics = {
        "extraction_time": datetime.now().isoformat(),
        "results_dir": str(results_dir),
    }
    
    # 尝试从不同来源提取指标
    try:
        # 从 results.results_dict 提取（训练结束时的指标）
        if hasattr(results, 'results_dict') and results.results_dict:
            metrics.update({
                "final_metrics": results.results_dict
            })
        
        # 从验证结果提取
        if hasattr(results, 'maps') and results.maps is not None:
            # mAP 指标
            if hasattr(results.maps, '__iter__'):
                metrics["mAP50-95"] = float(results.maps[0]) if len(results.maps) > 0 else None
            else:
                metrics["mAP50-95"] = float(results.maps)
        
        # 从 results.box 提取（如果是检测任务）
        if hasattr(results, 'box'):
            box_metrics = {}
            if hasattr(results.box, 'map'):
                box_metrics["mAP50-95"] = float(results.box.map)
            if hasattr(results.box, 'map50'):
                box_metrics["mAP50"] = float(results.box.map50)
            if hasattr(results.box, 'map75'):
                box_metrics["mAP75"] = float(results.box.map75)
            if hasattr(results.box, 'mp'):
                box_metrics["precision"] = float(results.box.mp)
            if hasattr(results.box, 'mr'):
                box_metrics["recall"] = float(results.box.mr)
            
            if box_metrics:
                metrics["box"] = box_metrics
        
        # 尝试读取 results.csv（如果存在）
        results_csv = Path(results_dir) / "results.csv"
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            last_row = df.iloc[-1].to_dict()
            
            # 提取关键指标
            key_metrics = {}
            metric_map = {
                "metrics/mAP50(B)": "mAP50",
                "metrics/mAP50-95(B)": "mAP50-95",
                "metrics/precision(B)": "precision",
                "metrics/recall(B)": "recall",
                "train/box_loss": "box_loss",
                "train/cls_loss": "cls_loss",
                "train/dfl_loss": "dfl_loss",
                "val/box_loss": "val_box_loss",
                "val/cls_loss": "val_cls_loss",
                "val/dfl_loss": "val_dfl_loss",
            }
            
            for csv_key, metric_key in metric_map.items():
                if csv_key in last_row:
                    key_metrics[metric_key] = float(last_row[csv_key])
            
            if key_metrics:
                metrics["from_csv"] = key_metrics
        
    except Exception as e:
        print(f"⚠️  提取指标时出现部分错误: {e}")
        metrics["extraction_error"] = str(e)
    
    return metrics


def save_model_weights(results_dir, snapshot_dir):
    """保存或链接模型权重文件"""
    results_dir = Path(results_dir)
    snapshot_dir = Path(snapshot_dir)
    weights_dir = snapshot_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("保存模型权重")
    print(f"{'='*80}")
    
    # 查找权重文件
    weights_source = results_dir / "weights"
    if not weights_source.exists():
        print(f"⚠️  权重目录不存在: {weights_source}")
        return
    
    # 复制 best.pt 和 last.pt
    for weight_file in ["best.pt", "last.pt"]:
        source = weights_source / weight_file
        target = weights_dir / weight_file
        
        if source.exists():
            shutil.copy2(source, target)
            size_mb = source.stat().st_size / (1024 * 1024)
            try:
                rel_path = target.relative_to(Path.cwd())
            except ValueError:
                rel_path = target
            print(f"✓ {weight_file} 已复制: {rel_path} ({size_mb:.2f} MB)")
        else:
            print(f"⚠️  {weight_file} 不存在")
    
    print(f"{'='*80}\n")


def run_training(exp_name, model_yaml, seed, train_cfg, **overrides):
    """运行一次训练实验"""
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "训练运行器启动" + " "*39 + "║")
    print("╚" + "="*78 + "╝\n")
    
    # 1. 设置随机种子
    print(f"{'='*80}")
    print(f"步骤 1/6: 设置随机种子")
    print(f"{'='*80}")
    set_seed(seed)
    print(f"✓ 随机种子已设置: {seed}\n")
    
    # 2. 创建快照目录
    snapshot_dir = Path("results/runs") / exp_name / f"seed{seed}"
    print(f"{'='*80}")
    print(f"步骤 2/6: 创建快照目录")
    print(f"{'='*80}")
    print(f"快照目录: {snapshot_dir}")
    
    # 如果目录已存在，询问是否覆盖
    if snapshot_dir.exists():
        print(f"⚠️  快照目录已存在: {snapshot_dir}")
        response = input("是否覆盖？(y/N): ").strip().lower()
        if response != 'y':
            print("❌ 训练已取消")
            return None
        shutil.rmtree(snapshot_dir)
        print("✓ 已删除旧快照")
    
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ 快照目录已创建\n")
    
    # 3. 加载训练配置
    print(f"{'='*80}")
    print(f"步骤 3/6: 加载训练配置")
    print(f"{'='*80}")
    train_config = load_config(train_cfg, validate=True)
    print(f"✓ 基础配置已加载: {train_cfg}")
    
    # 应用覆盖参数
    for key, value in overrides.items():
        if value is not None:
            train_config[key] = value
            print(f"  覆盖参数: {key} = {value}")
    
    # 设置必需参数
    train_config['model'] = model_yaml
    train_config['name'] = f"{exp_name}_seed{seed}"
    train_config['project'] = str(Path('results') / 'train')
    train_config['seed'] = seed
    train_config['exist_ok'] = True  # 允许覆盖
    
    print(f"✓ 配置参数已准备完成")
    print(f"  - 模型: {model_yaml}")
    print(f"  - 数据集: {train_config.get('data', 'N/A')}")
    print(f"  - 轮数: {train_config.get('epochs', 'N/A')}")
    print(f"  - 批次: {train_config.get('batch', 'N/A')}")
    print(f"  - 设备: {train_config.get('device', 'N/A')}")
    print()
    
    # 4. 创建实验快照
    print(f"{'='*80}")
    print(f"步骤 4/6: 创建实验快照")
    print(f"{'='*80}")
    env_json_path = "results/metadata/env.json"
    snapshot_files = create_experiment_snapshot(
        snapshot_dir, 
        model_yaml, 
        train_config,
        env_json_path
    )
    
    # 5. 开始训练
    print(f"{'='*80}")
    print(f"步骤 5/6: 开始训练")
    print(f"{'='*80}")
    print(f"实验名称: {exp_name}")
    print(f"随机种子: {seed}")
    print(f"模型配置: {model_yaml}")
    print(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    try:
        # 加载模型
        model = YOLO(model_yaml)
        print(f"✓ 模型已加载")
        
        # 开始训练
        results = model.train(**train_config)
        print(f"\n✓ 训练完成")
        
        # 获取训练结果目录（Ultralytics会自动添加runs/detect/前缀）
        # project='results/train', name='baseline_seed0'
        # 实际输出: runs/detect/results/train/baseline_seed0
        # 所以需要手动构建完整路径
        results_dir = Path('runs') / 'detect' / train_config['project'] / train_config['name']
        if not results_dir.exists():
            # 备选方案：尝试直接使用配置的路径
            results_dir = Path(train_config['project']) / train_config['name']
        print(f"✓ 结果目录: {results_dir}")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 保存错误信息
        error_log = snapshot_dir / "error.log"
        with open(error_log, "w", encoding="utf-8") as f:
            f.write(f"训练失败时间: {datetime.now().isoformat()}\n")
            f.write(f"错误信息: {str(e)}\n\n")
            f.write("详细堆栈:\n")
            traceback.print_exc(file=f)
        print(f"✓ 错误日志已保存: {error_log}")
        return None
    
    # 6. 提取指标并保存
    print(f"\n{'='*80}")
    print(f"步骤 6/6: 提取和保存指标")
    print(f"{'='*80}")
    
    # 提取指标
    metrics = extract_metrics(results, results_dir)
    
    # 保存指标
    metrics_file = snapshot_dir / "metrics_train.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    try:
        rel_path = metrics_file.relative_to(Path.cwd())
    except ValueError:
        rel_path = metrics_file
    print(f"✓ 训练指标已保存: {rel_path}")
    
    # 打印关键指标
    if "from_csv" in metrics:
        csv_metrics = metrics["from_csv"]
        print(f"\n关键指标:")
        if "mAP50" in csv_metrics:
            print(f"  mAP50: {csv_metrics['mAP50']:.4f}")
        if "mAP50-95" in csv_metrics:
            print(f"  mAP50-95: {csv_metrics['mAP50-95']:.4f}")
        if "precision" in csv_metrics:
            print(f"  Precision: {csv_metrics['precision']:.4f}")
        if "recall" in csv_metrics:
            print(f"  Recall: {csv_metrics['recall']:.4f}")
    
    # 保存模型权重
    save_model_weights(results_dir, snapshot_dir)
    
    # 创建符号链接或记录 Ultralytics 输出目录
    link_info = snapshot_dir / "ultralytics_output.txt"
    with open(link_info, "w", encoding="utf-8") as f:
        f.write(f"Ultralytics 训练输出目录:\n{results_dir}\n\n")
        f.write(f"权重文件已复制到:\n{snapshot_dir / 'weights'}\n")
    try:
        rel_path = link_info.relative_to(Path.cwd())
    except ValueError:
        rel_path = link_info
    print(f"✓ 输出目录信息已保存: {rel_path}")
    
    # 最终总结
    print(f"\n{'='*80}")
    print("训练完成总结")
    print(f"{'='*80}")
    print(f"实验名称: {exp_name}")
    print(f"随机种子: {seed}")
    try:
        snap_rel = snapshot_dir.relative_to(Path.cwd())
        res_rel = results_dir.relative_to(Path.cwd())
    except ValueError:
        snap_rel = snapshot_dir
        res_rel = results_dir
    print(f"快照目录: {snap_rel}")
    print(f"训练结果: {res_rel}")
    print(f"\n快照内容:")
    print(f"  - 模型配置: model_config.yaml")
    print(f"  - 训练配置: resolved_train.yaml")
    print(f"  - 环境信息: env.json")
    print(f"  - 元数据: metadata.json")
    print(f"  - 训练指标: metrics_train.json")
    print(f"  - 模型权重: weights/best.pt, weights/last.pt")
    print(f"{'='*80}\n")
    
    return {
        "snapshot_dir": snapshot_dir,
        "results_dir": results_dir,
        "metrics": metrics,
        "success": True,
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="严格可追踪的训练运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练 baseline 模型 (seed=0)
  python scripts/run_train_one.py \\
      --exp_name baseline \\
      --model_yaml ultralytics/cfg/models/v8/yolov8n_baseline.yaml \\
      --seed 0

  # 训练 ghost 模型，使用自定义数据集
  python scripts/run_train_one.py \\
      --exp_name ghost \\
      --model_yaml ultralytics/cfg/models/v8/yolov8n_ghost.yaml \\
      --seed 0 \\
      --data datasets/selfparts/data.yaml

  # 训练 ECA 模型，覆盖训练参数
  python scripts/run_train_one.py \\
      --exp_name eca \\
      --model_yaml ultralytics/cfg/models/v8/yolov8n_eca.yaml \\
      --seed 1 \\
      --epochs 50 \\
      --batch 16
        """
    )
    
    # 必需参数
    parser.add_argument("--exp_name", type=str, required=True,
                       help="实验名称 (例如: baseline, ghost, eca, p2)")
    parser.add_argument("--model_yaml", type=str, required=True,
                       help="模型配置文件路径")
    parser.add_argument("--seed", type=int, required=True,
                       help="随机种子")
    
    # 可选参数
    parser.add_argument("--train_cfg", type=str, default="experiments/base_train.yaml",
                       help="训练配置文件路径 (默认: experiments/base_train.yaml)")
    
    # 训练参数覆盖
    parser.add_argument("--data", type=str, help="数据集配置文件")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--batch", type=int, help="批次大小")
    parser.add_argument("--imgsz", type=int, help="图像尺寸")
    parser.add_argument("--device", type=str, help="设备 (0, 0,1, cpu)")
    parser.add_argument("--workers", type=int, help="数据加载线程数")
    parser.add_argument("--optimizer", type=str, help="优化器 (SGD, Adam, AdamW)")
    parser.add_argument("--lr0", type=float, help="初始学习率")
    
    args = parser.parse_args()
    
    # 收集覆盖参数
    overrides = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
    }
    
    # 运行训练
    result = run_training(
        exp_name=args.exp_name,
        model_yaml=args.model_yaml,
        seed=args.seed,
        train_cfg=args.train_cfg,
        **overrides
    )
    
    if result and result["success"]:
        print("🎉 训练运行完成，所有快照已保存！")
        return 0
    else:
        print("❌ 训练运行失败，请检查错误日志。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
