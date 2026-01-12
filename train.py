"""
YOLOv8-Small-Part 训练脚本
使用 uv run 执行：uv run train_fixed.py
"""
import os
import sys

# 确保使用本地修改的源码
sys.path.insert(0, os.path.abspath("ultralytics"))

from ultralytics import YOLO

def main():
    print("=" * 70)
    print("🚀 开始训练 YOLOv8-Small-Part 模型")
    print("=" * 70)
    
    # 检查数据集配置文件
    if not os.path.exists("parts.yaml"):
        print("❌ 错误：找不到数据集配置文件 parts.yaml")
        return
    
    # 加载自定义创新模型架构
    print("\n📋 加载模型配置: yolov8-small-part.yaml")
    model = YOLO("yolov8-small-part.yaml")
    
    print("\n✅ 模型加载成功！")
    print(f"   - EMA 注意力: 已集成")
    print(f"   - 检测尺度: P2/P3/P4/P5 四尺度")
    print(f"   - 模型参数: 3.4M")
    
    print("\n" + "=" * 70)
    print("🏋️  开始训练...")
    print("=" * 70 + "\n")
    
    # 开始训练
    results = model.train(
        data="parts.yaml",           # 数据集配置文件
        epochs=100,                  # 训练轮数
        imgsz=640,                   # 输入图像尺寸
        batch=16,                    # 批大小（显存不够改成 8 或 4）
        device=0,                    # GPU 设备（没有 GPU 用 'cpu'）
        workers=4,                   # 数据加载线程数
        name="SmallPart_v1",         # 实验名称
        save=True,                   # 保存检查点
        resume=False,                # 是否恢复训练
        augment=True,                # 数据增强
        
        # 针对小目标优化的参数
        box=7.5,                     # 边框损失权重（提高小目标定位精度）
        cls=0.5,                     # 分类损失权重
        
        # 其他有用的参数
        project="runs/detect",       # 保存路径
        exist_ok=False,              # 是否覆盖已有实验
        pretrained=False,            # 从头训练（不使用预训练权重）
        optimizer="auto",            # 优化器（auto/SGD/Adam/AdamW）
        verbose=True,                # 详细输出
        seed=0,                      # 随机种子
        deterministic=True,          # 确定性训练（可复现）
        single_cls=False,            # 是否单类别检测
        rect=False,                  # 矩形训练
        cos_lr=False,                # 余弦学习率衰减
        close_mosaic=10,             # 最后 N 轮关闭 Mosaic 增强
        amp=True,                    # 自动混合精度训练
        fraction=1.0,                # 使用的数据集比例
        profile=False,               # 性能分析
        freeze=None,                 # 冻结层（None 或 [0, 1, 2, ...]）
        lr0=0.01,                    # 初始学习率
        lrf=0.01,                    # 最终学习率（相对于 lr0）
        momentum=0.937,              # SGD 动量/Adam beta1
        weight_decay=0.0005,         # 权重衰减
        warmup_epochs=3.0,           # 学习率预热轮数
        warmup_momentum=0.8,         # 预热初始动量
        warmup_bias_lr=0.1,          # 预热初始偏置学习率
        patience=100,                # 早停耐心值（epochs）
        plots=True,                  # 保存训练图表
        
        # 数据增强参数（针对工业场景）
        hsv_h=0.015,                 # HSV 色调增强
        hsv_s=0.7,                   # HSV 饱和度增强
        hsv_v=0.4,                   # HSV 亮度增强
        degrees=0.0,                 # 旋转角度（工业零件固定方向可设为 0）
        translate=0.1,               # 平移
        scale=0.5,                   # 缩放
        shear=0.0,                   # 剪切
        perspective=0.0,             # 透视变换
        flipud=0.0,                  # 上下翻转概率
        fliplr=0.5,                  # 左右翻转概率
        mosaic=1.0,                  # Mosaic 增强概率
        mixup=0.0,                   # Mixup 增强概率
        copy_paste=0.0,              # Copy-Paste 增强概率
    )
    
    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    print("=" * 70)
    print(f"\n📊 训练结果保存在: runs/detect/SmallPart_v1/")
    print(f"   - 最佳模型: runs/detect/SmallPart_v1/weights/best.pt")
    print(f"   - 最后模型: runs/detect/SmallPart_v1/weights/last.pt")
    print(f"\n💡 推理测试:")
    print(f"   model = YOLO('runs/detect/SmallPart_v1/weights/best.pt')")
    print(f"   results = model('test_image.jpg')")
    print()

if __name__ == "__main__":
    main()
