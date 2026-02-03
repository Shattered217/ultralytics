"""
配置使用示例 (Configuration Usage Examples)

本文件展示如何在训练和评估脚本中使用统一的配置框架。
确保所有实验使用一致的配置，避免人为不一致。

注意：这是示例文件，实际训练请使用 ultralytics 的官方接口。
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.load_config import load_config, save_config
from scripts.set_determinism import set_seed


def example_train():
    """训练脚本示例"""
    
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLO 训练示例')
    
    # 配置文件
    parser.add_argument('--cfg', type=str, default='experiments/base_train.yaml',
                       help='训练配置文件路径')
    
    # 可覆盖的参数
    parser.add_argument('--data', type=str, help='数据集配置文件')
    parser.add_argument('--model', type=str, help='模型配置文件')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch', type=int, help='批次大小')
    parser.add_argument('--imgsz', type=int, help='图像尺寸')
    parser.add_argument('--device', type=str, help='设备（0, 0,1, cpu）')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--name', type=str, help='实验名称')
    
    args = parser.parse_args()
    
    # 2. 设置随机种子（第一步：确保可复现）
    set_seed(args.seed)
    print(f"✅ 随机种子已设置: {args.seed}")
    
    # 3. 加载基础配置
    config = load_config(args.cfg, validate=True)
    print(f"✅ 加载配置文件: {args.cfg}")
    
    # 4. 应用命令行覆盖
    if args.data:
        config['data'] = args.data
    if args.model:
        config['model'] = args.model
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch:
        config['batch'] = args.batch
    if args.imgsz:
        config['imgsz'] = args.imgsz
    if args.device:
        config['device'] = args.device
    if args.name:
        config['name'] = args.name
    
    # 5. 验证最终配置
    print("\n" + "="*80)
    print("最终训练配置")
    print("="*80)
    print(f"数据集: {config['data']}")
    print(f"模型: {config.get('model', 'yolov8n.yaml (默认)')}")
    print(f"轮数: {config['epochs']}")
    print(f"批次: {config['batch']}")
    print(f"图像尺寸: {config['imgsz']}")
    print(f"设备: {config['device']}")
    print(f"学习率: {config['lr0']}")
    print(f"优化器: {config['optimizer']}")
    print(f"种子: {args.seed}")
    print("="*80 + "\n")
    
    # 6. 保存配置到结果目录（重要：用于复现）
    output_dir = Path(config['project']) / config['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_save_path = output_dir / 'train_config.yaml'
    save_config(config, config_save_path)
    
    # 7. 开始训练（实际使用 ultralytics API）
    print("🚀 开始训练...")
    print("   (这是示例脚本，实际训练请使用 ultralytics 的 train 接口)")
    
    # 实际训练代码示例：
    # from ultralytics import YOLO
    # model = YOLO(config.get('model', 'yolov8n.yaml'))
    # results = model.train(**config)
    
    return config


def example_eval():
    """评估脚本示例"""
    
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLO 评估示例')
    
    # 配置文件
    parser.add_argument('--cfg', type=str, default='experiments/base_eval.yaml',
                       help='评估配置文件路径')
    
    # 必需参数
    parser.add_argument('--weights', type=str, required=True,
                       help='模型权重文件路径')
    
    # 可覆盖的参数
    parser.add_argument('--data', type=str, help='数据集配置文件')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'],
                       help='评估的数据划分')
    parser.add_argument('--imgsz', type=int, help='图像尺寸')
    parser.add_argument('--batch', type=int, help='批次大小')
    parser.add_argument('--conf', type=float, help='置信度阈值')
    parser.add_argument('--iou', type=float, help='IoU 阈值')
    parser.add_argument('--device', type=str, help='设备')
    parser.add_argument('--name', type=str, help='实验名称')
    
    args = parser.parse_args()
    
    # 2. 加载基础配置
    config = load_config(args.cfg, validate=True)
    print(f"✅ 加载配置文件: {args.cfg}")
    
    # 3. 应用命令行覆盖
    config['weights'] = args.weights
    if args.data:
        config['data'] = args.data
    if args.split:
        config['split'] = args.split
    if args.imgsz:
        config['imgsz'] = args.imgsz
    if args.batch:
        config['batch'] = args.batch
    if args.conf:
        config['conf'] = args.conf
    if args.iou:
        config['iou'] = args.iou
    if args.device:
        config['device'] = args.device
    if args.name:
        config['name'] = args.name
    
    # 4. 验证最终配置
    print("\n" + "="*80)
    print("最终评估配置")
    print("="*80)
    print(f"权重: {config['weights']}")
    print(f"数据集: {config['data']}")
    print(f"划分: {config['split']}")
    print(f"图像尺寸: {config['imgsz']}")
    print(f"批次: {config['batch']}")
    print(f"置信度阈值: {config['conf']}")
    print(f"IoU 阈值: {config['iou']}")
    print(f"设备: {config['device']}")
    print("="*80 + "\n")
    
    # 5. 保存配置到结果目录
    output_dir = Path(config['project']) / config['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_save_path = output_dir / 'eval_config.yaml'
    save_config(config, config_save_path)
    
    # 6. 开始评估
    print("🔍 开始评估...")
    print("   (这是示例脚本，实际评估请使用 ultralytics 的 val 接口)")
    
    # 实际评估代码示例：
    # from ultralytics import YOLO
    # model = YOLO(config['weights'])
    # results = model.val(**config)
    
    return config


def example_usage():
    """打印使用示例"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                   配置使用示例 (Configuration Usage)                         ║
╚════════════════════════════════════════════════════════════════════════════╝

1️⃣  基础训练（使用默认配置）
   python train.py --cfg experiments/base_train.yaml

2️⃣  覆盖参数训练
   python train.py \\
       --cfg experiments/base_train.yaml \\
       --data datasets/selfparts/data.yaml \\
       --epochs 200 \\
       --batch 16 \\
       --name E1_ghost_seed0 \\
       --seed 0

3️⃣  多种子训练（消融实验）
   for seed in 0 1 2; do
       python train.py \\
           --cfg experiments/base_train.yaml \\
           --data datasets/openparts/data.yaml \\
           --epochs 100 \\
           --name E0_baseline_seed${seed} \\
           --seed ${seed}
   done

4️⃣  基础评估
   python val.py \\
       --cfg experiments/base_eval.yaml \\
       --weights results/train/E0_baseline_seed0/weights/best.pt \\
       --split val

5️⃣  测试集最终评估
   python val.py \\
       --cfg experiments/base_eval.yaml \\
       --weights results/train/E0_baseline_seed0/weights/best.pt \\
       --data datasets/openparts/data.yaml \\
       --split test \\
       --name E0_baseline_test

6️⃣  Python 脚本中使用配置
   ```python
   from scripts.load_config import load_config
   from scripts.set_determinism import set_seed
   
   # 设置种子
   set_seed(0)
   
   # 加载配置
   config = load_config('experiments/base_train.yaml', 
                       overrides={'epochs': 200, 'batch': 16})
   
   # 使用配置
   from ultralytics import YOLO
   model = YOLO('yolov8n.yaml')
   results = model.train(**config)
   ```

7️⃣  查看配置内容
   python scripts/load_config.py experiments/base_train.yaml --print

8️⃣  验证配置有效性
   python scripts/load_config.py experiments/base_eval.yaml --validate

╔════════════════════════════════════════════════════════════════════════════╗
║                          重要原则 (Key Principles)                           ║
╚════════════════════════════════════════════════════════════════════════════╝

✅ 必须遵守：
   • 所有训练必须引用 base_train.yaml
   • 所有评估必须引用 base_eval.yaml
   • 不允许在脚本中硬编码超参数
   • 每次实验保存完整配置到结果目录
   • 使用相同种子的多次运行取平均值

❌ 禁止：
   • 在不同实验中使用不同的训练配置
   • 修改 base_*.yaml 后不同步所有实验
   • 不记录使用的配置参数
   • 不设置随机种子

📝 实验流程：
   1. 设置随机种子 (set_seed)
   2. 加载基础配置 (load_config)
   3. 应用实验特定覆盖（仅模型结构）
   4. 保存最终配置
   5. 开始训练/评估
   6. 记录结果

""")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            example_train()
        elif sys.argv[1] == 'eval':
            example_eval()
        else:
            print(f"未知命令: {sys.argv[1]}")
            print("使用方法: python examples_config_usage.py [train|eval]")
    else:
        example_usage()
