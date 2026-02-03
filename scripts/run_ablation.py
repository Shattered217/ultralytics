"""
消融实验自动化运行器

功能：
1. 读取实验计划 (ablation_plan.yaml)
2. 逐个运行 (实验, 种子) 组合
3. 每个组合执行：训练 → 评估(val+test) → 性能基准测试
4. 记录失败并支持断点续跑

使用方法：
    python scripts/run_ablation.py --plan experiments/ablation_plan.yaml
    python scripts/run_ablation.py --plan experiments/ablation_plan.yaml --dry-run
    python scripts/run_ablation.py --plan experiments/ablation_plan.yaml --exp baseline --seed 0

作者: 科研复现协议
"""

import sys
from pathlib import Path
import argparse
import yaml
import json
import subprocess
from datetime import datetime
from collections import defaultdict
import os

# Windows UTF-8 输出修复
if sys.platform == "win32":
    import codecs
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="消融实验自动化运行器")
    
    parser.add_argument("--plan", default="experiments/ablation_plan.yaml",
                       help="实验计划YAML文件")
    parser.add_argument("--dry-run", action="store_true",
                       help="预览执行计划但不实际运行")
    parser.add_argument("--exp", help="只运行指定实验（名称）")
    parser.add_argument("--seed", type=int, help="只运行指定种子")
    parser.add_argument("--skip-train", action="store_true",
                       help="跳过训练（仅运行评估和基准测试）")
    parser.add_argument("--skip-eval", action="store_true",
                       help="跳过评估（仅运行训练和基准测试）")
    parser.add_argument("--skip-benchmark", action="store_true",
                       help="跳过基准测试（仅运行训练和评估）")
    parser.add_argument("--force", action="store_true",
                       help="强制重新运行（忽略已有输出）")
    
    return parser.parse_args()


def load_plan(plan_file):
    """加载实验计划"""
    with open(plan_file, encoding='utf-8') as f:
        plan = yaml.safe_load(f)
    
    # 验证必需字段
    required_keys = ["experiments", "seeds", "train_cfg", "eval_cfg"]
    for key in required_keys:
        if key not in plan:
            raise ValueError(f"实验计划缺少必需字段: {key}")
    
    return plan


def check_experiment_exists(exp_name, seed, plan):
    """检查实验输出是否已存在"""
    results_dir = Path("results/runs") / exp_name / f"seed{seed}"
    
    exists = {
        "train": False,
        "eval_val": False,
        "eval_test": False,
        "benchmark": False,
    }
    
    # 检查训练输出
    best_weights = results_dir / "weights" / "best.pt"
    if best_weights.exists():
        exists["train"] = True
    
    # 检查评估输出
    eval_val = results_dir / "eval_val.json"
    if eval_val.exists():
        exists["eval_val"] = True
    
    eval_test = results_dir / "eval_test.json"
    if eval_test.exists():
        exists["eval_test"] = True
    
    # 检查基准测试输出
    bench_file = Path("results/bench") / f"{exp_name}_seed{seed}.json"
    if bench_file.exists():
        exists["benchmark"] = True
    
    return exists


def log_failure(log_file, exp_name, seed, stage, error_msg):
    """记录失败信息"""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    
    with open(log_path, "a", encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Stage: {stage}\n")
        f.write(f"Error: {error_msg}\n")
        f.write(f"{'='*80}\n")


def run_command(cmd, description, dry_run=False):
    """运行命令"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"命令: {' '.join(cmd)}")
    
    if dry_run:
        print("[DRY-RUN] 跳过实际执行")
        return True, ""
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ {description} 完成")
        return True, ""
    
    except subprocess.CalledProcessError as e:
        error_msg = f"命令执行失败，退出码: {e.returncode}"
        print(f"❌ {description} 失败")
        print(f"错误: {error_msg}")
        return False, error_msg
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ {description} 失败")
        print(f"错误: {error_msg}")
        return False, error_msg


def run_experiment(exp_config, seed, plan, args):
    """运行单个实验"""
    exp_name = exp_config["name"]
    model_yaml = exp_config["model_yaml"]
    
    print(f"\n{'#'*80}")
    print(f"# 实验: {exp_name} | 种子: {seed}")
    print(f"# 模型: {model_yaml}")
    print(f"{'#'*80}")
    
    # 检查已有输出
    exists = check_experiment_exists(exp_name, seed, plan)
    resume = plan.get("resume", True) and not args.force
    
    if resume:
        print(f"\n断点续跑检查:")
        print(f"  - 训练: {'✓ 已完成' if exists['train'] else '✗ 待运行'}")
        print(f"  - 评估(val): {'✓ 已完成' if exists['eval_val'] else '✗ 待运行'}")
        print(f"  - 评估(test): {'✓ 已完成' if exists['eval_test'] else '✗ 待运行'}")
        print(f"  - 基准测试: {'✓ 已完成' if exists['benchmark'] else '✗ 待运行'}")
    
    log_file = plan.get("log_file", "results/ablation_failures.log")
    
    # ========== 步骤 1: 训练 ==========
    if not args.skip_train:
        if resume and exists["train"]:
            print(f"\n{'='*80}")
            print(f"跳过训练（已存在输出）")
            print(f"{'='*80}")
        else:
            cmd = [
                "uv", "run",
                "scripts/run_train_one.py",
                "--exp_name", exp_name,
                "--model_yaml", model_yaml,
                "--seed", str(seed),
                "--train_cfg", plan["train_cfg"],
            ]
            
            # 添加训练参数覆盖
            if "train_overrides" in plan:
                overrides = plan["train_overrides"]
                if "epochs" in overrides:
                    cmd.extend(["--epochs", str(overrides["epochs"])])
                if "batch" in overrides:
                    cmd.extend(["--batch", str(overrides["batch"])])
                if "imgsz" in overrides:
                    cmd.extend(["--imgsz", str(overrides["imgsz"])])
                if "device" in overrides:
                    cmd.extend(["--device", str(overrides["device"])])
                if "workers" in overrides:
                    cmd.extend(["--workers", str(overrides["workers"])])
                if "data" in overrides:
                    cmd.extend(["--data", overrides["data"]])
            
            success, error = run_command(
                cmd,
                f"训练: {exp_name} (seed={seed})",
                args.dry_run
            )
            
            if not success:
                log_failure(log_file, exp_name, seed, "train", error)
                if plan.get("on_failure", "continue") == "stop":
                    return False
                return False
    
    # 检查训练输出
    weights_path = Path("results/runs") / exp_name / f"seed{seed}" / "weights" / "best.pt"
    if not args.dry_run and not weights_path.exists():
        error = f"训练权重不存在: {weights_path}"
        print(f"❌ {error}")
        log_failure(log_file, exp_name, seed, "train_check", error)
        return False
    
    # ========== 步骤 2: 评估 ==========
    if not args.skip_eval:
        eval_splits = plan.get("eval_splits", ["val", "test"])
        
        for split in eval_splits:
            skip_key = f"eval_{split}"
            if resume and exists.get(skip_key, False):
                print(f"\n{'='*80}")
                print(f"跳过评估 ({split})（已存在输出）")
                print(f"{'='*80}")
                continue
            
            cmd = [
                "uv", "run",
                "scripts/run_eval_one.py",
                "--exp_name", exp_name,
                "--weights", str(weights_path),
                "--seed", str(seed),
                "--split", split,
                "--eval_cfg", plan["eval_cfg"],
            ]
            
            success, error = run_command(
                cmd,
                f"评估 ({split}): {exp_name} (seed={seed})",
                args.dry_run
            )
            
            if not success:
                log_failure(log_file, exp_name, seed, f"eval_{split}", error)
                if plan.get("on_failure", "continue") == "stop":
                    return False
    
    # ========== 步骤 3: 性能基准测试 ==========
    if not args.skip_benchmark and plan.get("benchmark_enabled", True):
        if resume and exists["benchmark"]:
            print(f"\n{'='*80}")
            print(f"跳过基准测试（已存在输出）")
            print(f"{'='*80}")
        else:
            bench_cfg = plan.get("benchmark_config", {})
            
            cmd = [
                "uv", "run",
                "scripts/benchmark_model.py",
                "--weights", str(weights_path),
                "--imgsz", str(bench_cfg.get("imgsz", 640)),
                "--device", str(bench_cfg.get("device", 0)),
                "--warmup", str(bench_cfg.get("warmup", 50)),
                "--iters", str(bench_cfg.get("iters", 300)),
                "--batch", str(bench_cfg.get("batch", 1)),
            ]
            
            # 第一个实验创建基准列表，后续复用
            if bench_cfg.get("use_benchmark_list", True):
                benchmark_list = Path("results/bench/benchmark_list.txt")
                if benchmark_list.exists():
                    cmd.append("--use_benchmark_list")
            
            success, error = run_command(
                cmd,
                f"基准测试: {exp_name} (seed={seed})",
                args.dry_run
            )
            
            if not success:
                log_failure(log_file, exp_name, seed, "benchmark", error)
                # 基准测试失败不阻止整体流程
    
    print(f"\n{'#'*80}")
    print(f"# ✅ 实验完成: {exp_name} | 种子: {seed}")
    print(f"{'#'*80}\n")
    
    return True


def generate_summary(plan, results):
    """生成执行摘要"""
    print(f"\n{'='*80}")
    print(f"消融实验执行摘要")
    print(f"{'='*80}")
    
    total = len(results)
    success = sum(1 for r in results if r["success"])
    failed = total - success
    
    print(f"\n总任务数: {total}")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    
    if failed > 0:
        print(f"\n失败任务:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['exp_name']} (seed={r['seed']})")
        
        log_file = plan.get("log_file", "results/ablation_failures.log")
        print(f"\n详细错误日志: {log_file}")
    
    # 统计各实验的完成情况
    exp_stats = defaultdict(lambda: {"total": 0, "success": 0})
    for r in results:
        exp_name = r["exp_name"]
        exp_stats[exp_name]["total"] += 1
        if r["success"]:
            exp_stats[exp_name]["success"] += 1
    
    print(f"\n各实验完成情况:")
    for exp_name in sorted(exp_stats.keys()):
        stats = exp_stats[exp_name]
        print(f"  {exp_name}: {stats['success']}/{stats['total']}")
    
    print(f"\n{'='*80}\n")
    
    return success, failed


def main():
    """主函数"""
    args = parse_args()
    
    print(f"\n{'='*80}")
    print(f"消融实验自动化运行器")
    print(f"{'='*80}")
    print(f"实验计划: {args.plan}")
    
    if args.dry_run:
        print(f"模式: DRY-RUN（预览）")
    
    # 加载实验计划
    try:
        plan = load_plan(args.plan)
    except Exception as e:
        print(f"❌ 加载实验计划失败: {e}")
        return 1
    
    print(f"✓ 实验计划已加载")
    
    # 筛选实验
    experiments = plan["experiments"]
    if args.exp:
        experiments = [e for e in experiments if e["name"] == args.exp]
        if not experiments:
            print(f"❌ 未找到实验: {args.exp}")
            return 1
    
    # 只保留启用的实验
    experiments = [e for e in experiments if e.get("enabled", True)]
    
    # 筛选种子
    seeds = plan["seeds"]
    if args.seed is not None:
        if args.seed not in seeds:
            print(f"❌ 种子 {args.seed} 不在计划中: {seeds}")
            return 1
        seeds = [args.seed]
    
    print(f"\n执行计划:")
    print(f"  - 实验数: {len(experiments)}")
    print(f"  - 种子数: {len(seeds)}")
    print(f"  - 总任务数: {len(experiments) * len(seeds)}")
    
    print(f"\n实验列表:")
    for exp in experiments:
        print(f"  - {exp['name']}: {exp.get('description', 'N/A')}")
    
    print(f"\n随机种子: {seeds}")
    
    print(f"\n配置:")
    print(f"  - 训练配置: {plan['train_cfg']}")
    print(f"  - 评估配置: {plan['eval_cfg']}")
    print(f"  - 基准测试: {'启用' if plan.get('benchmark_enabled', True) else '禁用'}")
    print(f"  - 断点续跑: {'启用' if plan.get('resume', True) and not args.force else '禁用'}")
    print(f"  - 失败策略: {plan.get('on_failure', 'continue')}")
    
    if args.dry_run:
        print(f"\n[DRY-RUN] 这是预览模式，不会实际执行命令")
    
    # 确认执行
    if not args.dry_run:
        try:
            response = input("\n是否开始执行？(y/N): ")
            if response.lower() != 'y':
                print("已取消")
                return 0
        except KeyboardInterrupt:
            print("\n已取消")
            return 0
    
    # 执行实验
    results = []
    start_time = datetime.now()
    
    for exp in experiments:
        for seed in seeds:
            success = run_experiment(exp, seed, plan, args)
            results.append({
                "exp_name": exp["name"],
                "seed": seed,
                "success": success,
            })
            
            # 如果失败且策略是停止
            if not success and plan.get("on_failure", "continue") == "stop":
                print(f"\n❌ 遇到失败，停止执行（on_failure=stop）")
                break
        
        # 外层也要检查
        if not success and plan.get("on_failure", "continue") == "stop":
            break
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # 生成摘要
    success_count, failed_count = generate_summary(plan, results)
    
    print(f"总耗时: {duration}")
    print(f"{'='*80}\n")
    
    # 返回码
    if failed_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
