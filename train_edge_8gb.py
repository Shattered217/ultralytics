"""
GSE-YOLOv8训练脚本（8GB 显存）
"""

import os
import sys

# 添加本地 ultralytics 到路径（如果使用本地修改版本）
sys.path.insert(0, os.path.abspath("../ultralytics"))

from ultralytics import YOLO


def main():
    print("=" * 80)
    print("🚀 GSE-YOLOv8 训练 ")
    print("=" * 80)
    
    # 检查数据集配置
    if not os.path.exists("../parts.yaml"):
        print("❌ 错误：找不到数据集配置文件 parts.yaml")
        print("   请确保 parts.yaml 在项目根目录")
        return
    
    print("\n📋 加载模型配置: GSE-YOLO.yaml")
    model = YOLO("GSE-YOLO.yaml")
    
    print("\n" + "=" * 80)
    print("🏋️  开始训练...")
    print("=" * 80 + "\n")
    
    results = model.train(
        # ========== 数据集配置 ==========
        data="../parts.yaml",
        epochs=200,
        imgsz=640,
        batch=16,           # ⚠️ 降低到 16（从 32）
        device=0,
        workers=2,          # ⚠️ 降低到 2（从 4）
        
        # ========== 训练策略 ==========
        name="SmallPart_EdgeV4_8GB",
        save=True,
        resume=False,
        exist_ok=False,
        pretrained=False,
        
        # ========== 优化器配置 ==========
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # ========== 损失函数权重（强化小目标）==========
        box=10.0,
        cls=0.5,
        dfl=2.0,
        
        # ========== 数据增强（小目标专用）==========
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.15,
        erasing=0.4,
        
        degrees=15.0,
        translate=0.2,
        scale=0.9,
        shear=3.0,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.5,
        
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        
        auto_augment="randaugment",
        
        # ========== 训练控制 ==========
        close_mosaic=15,
        patience=100,
        
        amp=True,           # 混合精度训练（节省显存）
        fraction=1.0,
        profile=False,
        freeze=None,
        
        # ========== 验证配置 ==========
        val=True,
        plots=True,
        save_period=-1,
        
        # ========== 其他 ==========
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        multi_scale=False,
    )
    
    print("\n" + "=" * 80)
    print("✅ 训练完成！")
    print("=" * 80)
    print(f"\n📊 训练结果保存在: runs/detect/SmallPart_Edge_8GB/")
    print(f"   - 最佳模型: runs/detect/SmallPart_Edge_8GB/weights/best.pt")
    print(f"   - 最后模型: runs/detect/SmallPart_Edge_8GB/weights/last.pt")
    
    try:
        print(f"\n📈 性能分析:")
        print(f"   mAP50:    {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
        print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.3f}")
    except:
        pass
    
if __name__ == "__main__":
    main()
