"""
记录实验环境信息 (Record Experiment Environment)

该脚本自动收集并保存当前运行环境的详细信息，确保实验可复现性。
输出文件: results/metadata/env.json

使用方法:
    python scripts/record_env.py
"""

import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path


def get_uv_info():
    """获取 uv 环境信息"""
    import subprocess
    
    uv_info = {
        "is_uv_env": False,
        "uv_version": None,
        "venv_path": None,
        "uv_lock_exists": False,
    }
    
    # 检查是否在虚拟环境中
    if hasattr(sys, 'prefix') and sys.prefix != sys.base_prefix:
        uv_info["is_uv_env"] = True
        uv_info["venv_path"] = sys.prefix
    
    # 检查 VIRTUAL_ENV 环境变量
    if os.environ.get('VIRTUAL_ENV'):
        uv_info["is_uv_env"] = True
        uv_info["venv_path"] = os.environ.get('VIRTUAL_ENV')
    
    # 尝试获取 uv 版本
    try:
        result = subprocess.run(
            ['uv', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            uv_info["uv_version"] = result.stdout.strip()
    except Exception:
        pass
    
    # 检查 uv.lock 文件是否存在
    project_root = Path(__file__).parent.parent
    uv_lock = project_root / 'uv.lock'
    if uv_lock.exists():
        uv_info["uv_lock_exists"] = True
    
    return uv_info


def get_installed_packages_uv():
    """使用 uv 获取已安装的包列表"""
    import subprocess
    
    packages = {}
    
    try:
        # 使用 uv pip list 获取包列表
        result = subprocess.run(
            ['uv', 'pip', 'list', '--format', 'json'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            import json as json_module
            pkg_list = json_module.loads(result.stdout)
            for pkg in pkg_list:
                packages[pkg['name']] = pkg['version']
    except Exception:
        # Fallback 到标准方法
        pass
    
    return packages


def get_git_commit_hash():
    """获取当前 Git commit hash"""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent  # ultralytics 根目录
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "unknown (not a git repository or git not available)"
    except Exception as e:
        return f"error: {str(e)}"


def get_git_branch():
    """获取当前 Git 分支"""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "unknown"
    except Exception:
        return "unknown"


def get_ultralytics_version():
    """获取 Ultralytics 版本"""
    # 优先从 pyproject.toml 读取（本地开发环境）
    pyproject_path = Path(__file__).parent.parent / 'pyproject.toml'
    if pyproject_path.exists():
        try:
            with open(pyproject_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 简单解析 version 行
                for line in content.split('\n'):
                    if line.strip().startswith('version'):
                        # 匹配 version = "x.x.x" 格式
                        import re
                        match = re.search(r'version\s*=\s*["\']([^"\'\']+)["\']', line)
                        if match:
                            return match.group(1)
        except:
            pass
    
    # Fallback: 尝试导入 ultralytics 模块
    try:
        import ultralytics
        return ultralytics.__version__
    except (ImportError, AttributeError):
        return "dev (local)"


def get_python_packages():
    """获取关键 Python 包版本"""
    packages = {}
    key_packages = [
        'torch', 'torchvision', 'numpy', 'opencv-python', 
        'pillow', 'matplotlib', 'tqdm', 'pyyaml', 'scipy'
    ]
    
    # 首先尝试从 uv 获取所有包
    uv_packages = get_installed_packages_uv()
    
    for pkg_name in key_packages:
        # 先从 uv 列表查找
        if pkg_name in uv_packages:
            packages[pkg_name] = uv_packages[pkg_name]
            continue
        
        # Fallback 到直接导入
        try:
            if pkg_name == 'opencv-python':
                import cv2
                packages['opencv-python'] = cv2.__version__
            else:
                pkg = __import__(pkg_name.replace('-', '_'))
                packages[pkg_name] = pkg.__version__
        except ImportError:
            packages[pkg_name] = "not installed"
        except AttributeError:
            packages[pkg_name] = "unknown"
    
    return packages


def record_environment():
    """收集并记录环境信息"""
    
    # 收集 uv 环境信息
    uv_info = get_uv_info()
    
    # 获取包列表
    packages = get_python_packages()
    
    # 收集环境信息
    env_info = {
        "timestamp": datetime.now().isoformat(),
        "uv": uv_info,
        "system": {
            "os": platform.system(),
            "os_version": platform.version(),
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
        },
        "ultralytics": {
            "version": get_ultralytics_version(),
        },
        "git": {
            "commit_hash": get_git_commit_hash(),
            "branch": get_git_branch(),
        },
        "packages": packages,
    }
    
    return env_info


def print_environment(env_info):
    """打印环境信息到控制台"""
    print("=" * 80)
    print("实验环境信息 (Experiment Environment)")
    print("=" * 80)
    
    print(f"\n📅 时间: {env_info['timestamp']}")
    
    # 打印 uv 环境信息
    print(f"\n📦 UV 环境管理器:")
    if env_info['uv']['uv_version']:
        print(f"  版本: {env_info['uv']['uv_version']}")
    if env_info['uv']['uv_lock_exists']:
        print(f"  ✓ uv.lock 文件存在")
    if env_info['uv']['is_uv_env']:
        print(f"  ✓ 当前运行在虚拟环境")
        if env_info['uv']['venv_path']:
            print(f"  路径: {env_info['uv']['venv_path']}")
    
    print(f"\n🐍 Python 版本: {env_info['system']['python_version']}")
    
    print(f"\n🚀 Ultralytics 版本: {env_info['ultralytics']['version']}")
    
    print(f"\n📦 Git:")
    print(f"  分支: {env_info['git']['branch']}")
    print(f"  Commit: {env_info['git']['commit_hash'][:8]}...")
    
    print(f"\n📚 关键依赖包:")
    # 按重要性排序显示
    priority_pkgs = ['torch', 'torchvision', 'numpy', 'opencv-python']
    for pkg in priority_pkgs:
        if pkg in env_info['packages']:
            version = env_info['packages'][pkg]
            # 高亮 PyTorch 版本（包含 CUDA 信息）
            if pkg == 'torch' and '+cu' in version:
                cuda_ver = version.split('+cu')[1]
                print(f"  ✓ {pkg}: {version.split('+')[0]} (CUDA {cuda_ver})")
            else:
                print(f"  ✓ {pkg}: {version}")
    
    # 显示其他包
    other_pkgs = [k for k in env_info['packages'].keys() if k not in priority_pkgs]
    if other_pkgs:
        print(f"  其他: {', '.join([f'{k} ({env_info['packages'][k]})' for k in other_pkgs])}")
    
    print(f"\n💾 操作系统: {env_info['system']['os']} {env_info['system']['platform']}")
    
    print("\n" + "=" * 80)


def save_environment(env_info, output_path):
    """保存环境信息到 JSON 文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(env_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 环境信息已保存到: {output_path}")
    print(f"   文件大小: {output_path.stat().st_size} bytes")


def main():
    """主函数"""
    # 收集环境信息
    print("正在收集环境信息...")
    env_info = record_environment()
    
    # 打印到控制台
    print_environment(env_info)
    
    # 保存到文件
    output_path = Path(__file__).parent.parent / 'results' / 'metadata' / 'env.json'
    save_environment(env_info, output_path)
    
    print("\n✨ 完成！环境信息已记录。")


if __name__ == "__main__":
    main()
