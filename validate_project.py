#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Project Validation Script
========================
验证项目结构和关键组件是否正常工作
"""

import os
import sys
from pathlib import Path

def check_file_exists(path, description):
    """检查文件是否存在"""
    if Path(path).exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} (missing)")
        return False

def check_directory_structure():
    """检查目录结构"""
    print("=== 检查目录结构 ===")
    
    required_dirs = [
        ("models/", "模型实现目录"),
        ("scripts/", "脚本目录"),
        ("tests/", "测试目录"),
        (".github/workflows/", "CI/CD配置目录"),
        ("data/", "数据目录"),
        ("blend/", "融合结果目录")
    ]
    
    optional_dirs = [
        ("outputs/", "输出目录（可选）")
    ]
    
    all_good = True
    for dir_path, desc in required_dirs:
        if not check_file_exists(dir_path, desc):
            all_good = False
    
    # 检查可选目录
    for dir_path, desc in optional_dirs:
        check_file_exists(dir_path, desc)
    
    return all_good

def check_core_files():
    """检查核心文件"""
    print("\n=== 检查核心文件 ===")
    
    required_files = [
        ("README.md", "项目文档"),
        ("requirements.txt", "依赖管理"),
        ("setup.py", "项目打包配置"),
        ("Makefile", "项目管理"),
        ("Dockerfile", "容器化配置"),
        ("LICENSE", "许可证"),
        (".gitignore", "Git忽略配置"),
        ("models/__init__.py", "模型包初始化"),
        ("scripts/__init__.py", "脚本包初始化"),
        ("tests/__init__.py", "测试包初始化")
    ]
    
    all_good = True
    for file_path, desc in required_files:
        if not check_file_exists(file_path, desc):
            all_good = False
    
    return all_good

def check_scripts():
    """检查脚本文件"""
    print("\n=== 检查脚本文件 ===")
    
    scripts = [
        ("scripts/feature_engineering_v1.py", "FE1特征工程"),
        ("scripts/feature_engineering_v2.py", "FE2特征工程"),
        ("scripts/feature_engineering_v3.py", "FE3特征工程"),
        ("scripts/train_models.py", "模型训练脚本"),
        ("scripts/blend.py", "模型融合脚本")
    ]
    
    all_good = True
    for script_path, desc in scripts:
        if not check_file_exists(script_path, desc):
            all_good = False
    
    return all_good

def check_models():
    """检查模型实现"""
    print("\n=== 检查模型实现 ===")
    
    models = [
        ("models/base_model.py", "基础模型类"),
        ("models/lightgbm_model.py", "LightGBM模型"),
        ("models/xgboost_model.py", "XGBoost模型"),
        ("models/catboost_model.py", "CatBoost模型"),
        ("models/linear_model.py", "线性模型")
    ]
    
    all_good = True
    for model_path, desc in models:
        if not check_file_exists(model_path, desc):
            all_good = False
    
    return all_good

def test_imports():
    """测试导入"""
    print("\n=== 测试导入 ===")
    
    # 测试基础脚本导入
    try:
        sys.path.append(str(Path(__file__).parent))
        import scripts.blend as blend
        print("✓ scripts.blend 导入成功")
    except Exception as e:
        print(f"✗ scripts.blend 导入失败: {e}")
        return False
    
    try:
        import scripts.feature_engineering_v1 as fe1
        print("✓ scripts.feature_engineering_v1 导入成功")
    except Exception as e:
        print(f"✗ scripts.feature_engineering_v1 导入失败: {e}")
        return False
    
    # 测试模型导入（可能因为缺少ML库而失败，这是正常的）
    try:
        from models import BaseModel
        print("✓ models.BaseModel 导入成功")
    except ImportError as e:
        print(f"⚠ models.BaseModel 导入失败（需要ML库依赖）: {e}")
    except Exception as e:
        print(f"✗ models.BaseModel 导入失败: {e}")
        return False
    
    return True

def check_ci_config():
    """检查CI配置"""
    print("\n=== 检查CI配置 ===")
    
    ci_files = [
        (".github/workflows/ci.yml", "GitHub Actions CI配置")
    ]
    
    all_good = True
    for file_path, desc in ci_files:
        if not check_file_exists(file_path, desc):
            all_good = False
    
    return all_good

def main():
    """主函数"""
    print("项目验证开始...")
    print("=" * 50)
    
    checks = [
        check_directory_structure,
        check_core_files,
        check_scripts,
        check_models,
        check_ci_config,
        test_imports
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"检查失败: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("验证结果汇总:")
    
    if all(results):
        print("🎉 所有检查通过！项目结构完整且正确。")
        return True
    else:
        print("❌ 部分检查失败，请检查上述错误。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
