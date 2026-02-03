"""
配置加载工具 (Configuration Loader)

该模块提供统一的配置加载接口，支持从 YAML 文件读取配置并允许通过命令行参数覆盖。
确保所有实验使用一致的配置基准。

功能：
1. 加载基础配置文件（base_train.yaml / base_eval.yaml）
2. 支持命令行参数覆盖配置项
3. 验证配置参数的有效性
4. 合并多个配置文件
5. 保存最终使用的配置

使用方法:
    from scripts.load_config import load_config, merge_configs, save_config
    
    # 基本使用
    config = load_config('experiments/base_train.yaml')
    
    # 命令行覆盖
    config = load_config('experiments/base_train.yaml', 
                        overrides={'epochs': 200, 'batch': 16})
    
    # 合并配置（实验特定配置 + 基础配置）
    config = merge_configs('experiments/base_train.yaml', 
                          'experiments/ablation_E1_ghost.yaml')
"""

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def load_yaml(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载 YAML 配置文件
    
    Args:
        yaml_path: YAML 文件路径
    
    Returns:
        dict: 配置字典
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理空文件
    if config is None:
        config = {}
    
    return config


def save_yaml(config: Dict[str, Any], yaml_path: Union[str, Path]) -> None:
    """
    保存配置到 YAML 文件
    
    Args:
        config: 配置字典
        yaml_path: 输出文件路径
    """
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def parse_value(value: str) -> Any:
    """
    解析字符串值为合适的 Python 类型
    
    Args:
        value: 字符串值
    
    Returns:
        解析后的值（int, float, bool, str 等）
    """
    # 处理 None
    if value.lower() in ['none', 'null']:
        return None
    
    # 处理布尔值
    if value.lower() in ['true', 'yes', 'on']:
        return True
    if value.lower() in ['false', 'no', 'off']:
        return False
    
    # 尝试解析为数字
    try:
        # 整数
        if '.' not in value and 'e' not in value.lower():
            return int(value)
        # 浮点数
        return float(value)
    except ValueError:
        pass
    
    # 返回字符串
    return value


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    应用覆盖参数到配置字典
    
    Args:
        config: 原始配置
        overrides: 覆盖参数字典
    
    Returns:
        dict: 更新后的配置
    """
    config = copy.deepcopy(config)
    
    for key, value in overrides.items():
        # 处理嵌套键（例如 'optimizer.lr' -> config['optimizer']['lr']）
        if '.' in key:
            keys = key.split('.')
            target = config
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = value
        else:
            config[key] = value
    
    return config


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
    validate: bool = True
) -> Dict[str, Any]:
    """
    加载配置文件并应用覆盖参数
    
    Args:
        config_path: 配置文件路径
        overrides: 覆盖参数字典（可选）
        validate: 是否验证配置（可选）
    
    Returns:
        dict: 最终配置
    
    Example:
        >>> config = load_config('experiments/base_train.yaml')
        >>> config = load_config('experiments/base_train.yaml', 
        ...                     overrides={'epochs': 200, 'batch': 16})
    """
    # 加载基础配置
    config = load_yaml(config_path)
    
    # 应用覆盖
    if overrides:
        config = apply_overrides(config, overrides)
    
    # 验证配置
    if validate:
        validate_config(config)
    
    return config


def merge_configs(
    base_config_path: Union[str, Path],
    override_config_path: Union[str, Path],
    extra_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    合并多个配置文件
    
    优先级：extra_overrides > override_config > base_config
    
    Args:
        base_config_path: 基础配置文件路径（如 base_train.yaml）
        override_config_path: 覆盖配置文件路径（如实验特定配置）
        extra_overrides: 额外的覆盖参数（可选）
    
    Returns:
        dict: 合并后的配置
    
    Example:
        >>> config = merge_configs('experiments/base_train.yaml',
        ...                       'experiments/ablation_E1_ghost.yaml',
        ...                       {'epochs': 200})
    """
    # 加载基础配置
    base_config = load_yaml(base_config_path)
    
    # 加载覆盖配置
    override_config = load_yaml(override_config_path)
    
    # 合并配置（深度合并）
    config = deep_merge(base_config, override_config)
    
    # 应用额外覆盖
    if extra_overrides:
        config = apply_overrides(config, extra_overrides)
    
    return config


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两个字典
    
    Args:
        base: 基础字典
        override: 覆盖字典
    
    Returns:
        dict: 合并后的字典
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并嵌套字典
            result[key] = deep_merge(result[key], value)
        else:
            # 直接覆盖
            result[key] = value
    
    return result


def validate_config(config: Dict[str, Any]) -> None:
    """
    验证配置参数的有效性
    
    Args:
        config: 配置字典
    
    Raises:
        ValueError: 如果配置无效
    """
    # 验证必需字段（训练配置）
    if config.get('mode') == 'train':
        required_train_fields = ['data', 'imgsz', 'epochs', 'batch']
        for field in required_train_fields:
            if field not in config:
                raise ValueError(f"训练配置缺少必需字段: {field}")
        
        # 验证数值范围
        if config['epochs'] <= 0:
            raise ValueError(f"epochs 必须为正数，当前值: {config['epochs']}")
        
        if config['imgsz'] <= 0:
            raise ValueError(f"imgsz 必须为正数，当前值: {config['imgsz']}")
        
        # 验证批次大小
        if config['batch'] != 'auto' and config['batch'] != -1:
            if not isinstance(config['batch'], int) or config['batch'] <= 0:
                raise ValueError(f"batch 必须为正整数或 'auto'/-1，当前值: {config['batch']}")
    
    # 验证必需字段（评估配置）
    if config.get('mode') == 'val':
        required_val_fields = ['data', 'imgsz', 'conf', 'iou']
        for field in required_val_fields:
            if field not in config:
                raise ValueError(f"评估配置缺少必需字段: {field}")
        
        # 验证阈值范围
        if not 0 <= config['conf'] <= 1:
            raise ValueError(f"conf 必须在 [0, 1] 范围内，当前值: {config['conf']}")
        
        if not 0 <= config['iou'] <= 1:
            raise ValueError(f"iou 必须在 [0, 1] 范围内，当前值: {config['iou']}")


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        output_path: 输出文件路径
    
    Example:
        >>> config = load_config('experiments/base_train.yaml')
        >>> save_config(config, 'results/train/exp/config.yaml')
    """
    save_yaml(config, output_path)
    print(f"✅ 配置已保存到: {output_path}")


def parse_args_overrides(args: argparse.Namespace, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    从 argparse 命名空间提取覆盖参数
    
    Args:
        args: argparse.Namespace 对象
        exclude: 要排除的参数列表（可选）
    
    Returns:
        dict: 覆盖参数字典
    
    Example:
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('--epochs', type=int)
        >>> parser.add_argument('--batch', type=int)
        >>> args = parser.parse_args(['--epochs', '200', '--batch', '16'])
        >>> overrides = parse_args_overrides(args)
        >>> # {'epochs': 200, 'batch': 16}
    """
    exclude = exclude or []
    overrides = {}
    
    for key, value in vars(args).items():
        if key not in exclude and value is not None:
            overrides[key] = value
    
    return overrides


def print_config(config: Dict[str, Any], title: str = "配置信息") -> None:
    """
    打印配置信息到控制台
    
    Args:
        config: 配置字典
        title: 标题
    """
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    
    for key, value in sorted(config.items()):
        if isinstance(value, dict):
            print(f"\n{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    
    print("=" * 80 + "\n")


# ============================================================================
# 命令行工具 (CLI Tool)
# ============================================================================

def main():
    """命令行工具：加载和验证配置文件"""
    parser = argparse.ArgumentParser(description='配置加载工具')
    parser.add_argument('config', type=str, help='配置文件路径')
    parser.add_argument('--override', nargs='+', help='覆盖参数（格式：key=value）')
    parser.add_argument('--validate', action='store_true', help='验证配置')
    parser.add_argument('--print', action='store_true', help='打印配置')
    parser.add_argument('--save', type=str, help='保存配置到文件')
    
    args = parser.parse_args()
    
    # 解析覆盖参数
    overrides = {}
    if args.override:
        for item in args.override:
            if '=' not in item:
                print(f"⚠️  忽略无效的覆盖参数: {item}")
                continue
            key, value = item.split('=', 1)
            overrides[key] = parse_value(value)
    
    # 加载配置
    try:
        config = load_config(args.config, overrides=overrides, validate=args.validate)
        print(f"✅ 成功加载配置: {args.config}")
        
        if overrides:
            print(f"   已应用 {len(overrides)} 个覆盖参数")
        
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        return
    
    # 打印配置
    if args.print:
        print_config(config)
    
    # 保存配置
    if args.save:
        save_config(config, args.save)


if __name__ == "__main__":
    main()
