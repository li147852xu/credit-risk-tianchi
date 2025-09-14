#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified Model Training Script
============================
统一的模型训练脚本，支持多种模型和配置
"""

import argparse
import json
from pathlib import Path
from models import (
    train_lightgbm_model, train_xgboost_model, train_catboost_model, train_linear_model,
    get_lightgbm_configs, get_xgboost_configs, get_catboost_configs, get_linear_configs
)


def get_all_configs():
    """获取所有可用的模型配置"""
    all_configs = {}
    
    # LightGBM配置
    lgb_configs = get_lightgbm_configs()
    for name, config in lgb_configs.items():
        all_configs[name] = {'model_type': 'lightgbm', 'config': config}
    
    # XGBoost配置
    xgb_configs = get_xgboost_configs()
    for name, config in xgb_configs.items():
        all_configs[name] = {'model_type': 'xgboost', 'config': config}
    
    # CatBoost配置
    cat_configs = get_catboost_configs()
    for name, config in cat_configs.items():
        all_configs[name] = {'model_type': 'catboost', 'config': config}
    
    # 线性模型配置
    linear_configs = get_linear_configs()
    for name, config in linear_configs.items():
        all_configs[name] = {'model_type': 'linear', 'config': config}
    
    return all_configs


def train_single_model(model_name, cache_dir, output_dir, **kwargs):
    """训练单个模型"""
    all_configs = get_all_configs()
    
    if model_name not in all_configs:
        available = list(all_configs.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    model_info = all_configs[model_name]
    model_type = model_info['model_type']
    
    print(f"Training {model_name} ({model_type})...")
    
    # 根据模型类型选择训练函数
    if model_type == 'lightgbm':
        return train_lightgbm_model(model_name, cache_dir, output_dir, **kwargs)
    elif model_type == 'xgboost':
        return train_xgboost_model(model_name, cache_dir, output_dir, **kwargs)
    elif model_type == 'catboost':
        return train_catboost_model(model_name, cache_dir, output_dir, **kwargs)
    elif model_type == 'linear':
        return train_linear_model(model_name, cache_dir, output_dir, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_multiple_models(model_names, cache_dir, output_dir, **kwargs):
    """训练多个模型"""
    results = {}
    
    for model_name in model_names:
        try:
            model, cv_results = train_single_model(model_name, cache_dir, output_dir, **kwargs)
            results[model_name] = {
                'cv_auc': cv_results['cv_auc'],
                'fold_results': cv_results['fold_results'],
                'status': 'success'
            }
            print(f"✓ {model_name}: AUC = {cv_results['cv_auc']:.6f}")
        except Exception as e:
            results[model_name] = {
                'error': str(e),
                'status': 'failed'
            }
            print(f"✗ {model_name}: Failed - {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train machine learning models for credit risk prediction")
    
    # 基本参数
    parser.add_argument("--model", type=str, help="Model name to train")
    parser.add_argument("--models", type=str, nargs='+', help="Multiple model names to train")
    parser.add_argument("--cache_dir", type=str, default="data/processed_v1", help="Cache directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    
    # 训练参数
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--random_state", type=int, default=2025, help="Random state")
    parser.add_argument("--save_models", action="store_true", default=True, help="Save trained models")
    parser.add_argument("--save_predictions", action="store_true", default=True, help="Save predictions")
    
    # 模型特定参数
    parser.add_argument("--num_boost_round", type=int, help="Number of boosting rounds")
    parser.add_argument("--early_stopping_rounds", type=int, help="Early stopping rounds")
    
    # 其他选项
    parser.add_argument("--list_models", action="store_true", help="List all available models")
    parser.add_argument("--config_file", type=str, help="JSON config file with model parameters")
    
    args = parser.parse_args()
    
    # 列出所有可用模型
    if args.list_models:
        all_configs = get_all_configs()
        print("Available models:")
        for name, info in all_configs.items():
            print(f"  {name} ({info['model_type']})")
        return
    
    # 加载配置文件
    config_overrides = {}
    if args.config_file:
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config_overrides = json.load(f)
    
    # 准备训练参数
    train_kwargs = {
        'n_folds': args.n_folds,
        'random_state': args.random_state,
        'save_models': args.save_models,
        'save_predictions': args.save_predictions
    }
    
    # 添加模型特定参数
    if args.num_boost_round:
        train_kwargs['num_boost_round'] = args.num_boost_round
    if args.early_stopping_rounds:
        train_kwargs['early_stopping_rounds'] = args.early_stopping_rounds
    
    # 合并配置文件参数
    train_kwargs.update(config_overrides)
    
    # 确定要训练的模型
    if args.model:
        models_to_train = [args.model]
    elif args.models:
        models_to_train = args.models
    else:
        print("Error: Please specify --model or --models")
        return
    
    # 训练模型
    print(f"Training models: {models_to_train}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"CV folds: {args.n_folds}")
    print()
    
    results = train_multiple_models(models_to_train, args.cache_dir, args.output_dir, **train_kwargs)
    
    # 保存训练结果摘要
    summary_path = Path(args.output_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nTraining summary saved to: {summary_path}")
    
    # 显示结果摘要
    print("\n=== Training Results ===")
    successful = [name for name, result in results.items() if result['status'] == 'success']
    failed = [name for name, result in results.items() if result['status'] == 'failed']
    
    if successful:
        print("Successful models:")
        for name in successful:
            auc = results[name]['cv_auc']
            print(f"  {name}: AUC = {auc:.6f}")
    
    if failed:
        print("\nFailed models:")
        for name in failed:
            error = results[name]['error']
            print(f"  {name}: {error}")


if __name__ == "__main__":
    main()
