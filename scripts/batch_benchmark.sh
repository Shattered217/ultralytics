#!/bin/bash
# 批量模型性能基准测试
# 用法: bash scripts/batch_benchmark.sh

set -e  # 遇到错误立即退出

echo "================================================"
echo "批量模型性能基准测试"
echo "================================================"

# 配置
DEVICE="0"
IMGSZ=640
WARMUP=50
ITERS=300
BATCH=1
SEED=42
BENCHMARK_SIZE=200

# 模型列表（实验名称）
EXPERIMENTS=(
    "baseline"
    "ghost"
    "eca"
    "p2"
    "ghost_eca"
    "ghost_eca_p2"
)

# 种子（如果有多个种子训练的模型）
SEED_ID=0

# 输出目录
OUTPUT_DIR="results/bench"
mkdir -p $OUTPUT_DIR

echo ""
echo "配置:"
echo "  - 设备: GPU $DEVICE"
echo "  - 输入尺寸: ${IMGSZ}x${IMGSZ}"
echo "  - 预热: $WARMUP 次"
echo "  - 测试: $ITERS 次"
echo "  - Batch Size: $BATCH"
echo "  - 基准图像数: $BENCHMARK_SIZE"
echo "  - 模型数量: ${#EXPERIMENTS[@]}"
echo ""

# 第一个模型：创建基准列表
FIRST_EXP=${EXPERIMENTS[0]}
FIRST_WEIGHTS="results/runs/${FIRST_EXP}/seed${SEED_ID}/weights/best.pt"

if [ ! -f "$FIRST_WEIGHTS" ]; then
    echo "❌ 错误：权重文件不存在: $FIRST_WEIGHTS"
    echo "请先训练模型或调整路径"
    exit 1
fi

echo "================================================"
echo "步骤 1: 测试第一个模型并创建基准列表"
echo "================================================"
echo "模型: $FIRST_EXP"
echo "权重: $FIRST_WEIGHTS"
echo ""

python scripts/benchmark_model.py \
    --weights "$FIRST_WEIGHTS" \
    --imgsz $IMGSZ \
    --device $DEVICE \
    --warmup $WARMUP \
    --iters $ITERS \
    --batch $BATCH \
    --benchmark_size $BENCHMARK_SIZE \
    --seed $SEED

if [ $? -ne 0 ]; then
    echo "❌ 第一个模型测试失败"
    exit 1
fi

echo "✅ 第一个模型测试完成"
echo ""

# 后续模型：复用基准列表
echo "================================================"
echo "步骤 2: 测试其余模型（复用基准列表）"
echo "================================================"

SUCCESS_COUNT=1
FAIL_COUNT=0

for EXP in "${EXPERIMENTS[@]:1}"; do
    WEIGHTS="results/runs/${EXP}/seed${SEED_ID}/weights/best.pt"
    
    echo ""
    echo "----------------------------------------"
    echo "模型: $EXP"
    echo "权重: $WEIGHTS"
    echo "----------------------------------------"
    
    if [ ! -f "$WEIGHTS" ]; then
        echo "⚠️  警告：权重文件不存在，跳过"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
    python scripts/benchmark_model.py \
        --weights "$WEIGHTS" \
        --imgsz $IMGSZ \
        --device $DEVICE \
        --warmup $WARMUP \
        --iters $ITERS \
        --batch $BATCH \
        --use_benchmark_list
    
    if [ $? -eq 0 ]; then
        echo "✅ $EXP 测试完成"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ $EXP 测试失败"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
echo "================================================"
echo "基准测试总结"
echo "================================================"
echo "总模型数: ${#EXPERIMENTS[@]}"
echo "成功: $SUCCESS_COUNT"
echo "失败: $FAIL_COUNT"
echo ""
echo "结果保存在: $OUTPUT_DIR/"
echo ""

# 列出所有结果文件
echo "生成的结果文件:"
ls -lh $OUTPUT_DIR/*.json 2>/dev/null || echo "  无JSON文件"

echo ""
echo "可以使用以下命令查看结果:"
echo "  cat $OUTPUT_DIR/baseline.json"
echo "  python scripts/compare_benchmarks.py"
echo ""
echo "================================================"
echo "批量测试完成！"
echo "================================================"
