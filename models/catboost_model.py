#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CatBoost Model Implementation
============================
CatBoost模型的实现，支持多种参数配置
"""

import os
import numpy as np
import pandas as pd
import catboost as cb
from pathlib import Path
from .base_model import BaseModel


class CatBoostModel(BaseModel):
    """CatBoost模型实现"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_params = self._get_default_params()
        self.model_params.update(config.get('catboost_params', {}))
        
    def _get_default_params(self):
        """获取默认CatBoost参数"""
        return {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'task_type': 'CPU',
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 5.0,
            'bagging_temperature': 1.0,
            'random_strength': 1.0,
            'one_hot_max_size': 2,
            'leaf_estimation_method': 'Newton',
            'bootstrap_type': 'Bayesian',
            'sampling_frequency': 'PerTree',
            'random_seed': self.config.get('random_state', 2025),
            'verbose': False,
            'thread_count': os.cpu_count() or 4
        }
    
    def train_single_fold(self, X_train, y_train, X_val, y_val, fold_idx):
        """训练单个fold的CatBoost模型"""
        # 确定类别特征索引
        cat_features = self.config.get('cat_features', [])
        cat_feature_indices = [i for i, col in enumerate(X_train.columns) if col in cat_features]
        
        # 创建数据集
        train_dataset = cb.Pool(
            X_train, 
            y_train,
            cat_features=cat_feature_indices
        )
        
        val_dataset = cb.Pool(
            X_val,
            y_val,
            cat_features=cat_feature_indices
        )
        
        # 训练参数
        num_boost_round = self.config.get('num_boost_round', 3000)
        early_stopping_rounds = self.config.get('early_stopping_rounds', 200)
        
        # 训练模型
        model = cb.CatBoost(
            **self.model_params,
            iterations=num_boost_round,
            early_stopping_rounds=early_stopping_rounds
        )
        
        model.fit(
            train_dataset,
            eval_set=val_dataset,
            verbose_eval=False
        )
        
        return model
    
    def predict_single_fold(self, model, X):
        """使用单个CatBoost模型进行预测"""
        return model.predict_proba(X)[:, 1]
    
    def _save_model_implementation(self, model, path):
        """保存CatBoost模型"""
        model.save_model(str(path))
    
    def get_feature_importance(self, importance_type='PredictionValuesChange'):
        """获取特征重要性"""
        if not self.models:
            return None
            
        importance_data = []
        for i, model in enumerate(self.models):
            importance = model.get_feature_importance(type=importance_type)
            feature_names = model.feature_names_
            
            for feat_name, imp in zip(feature_names, importance):
                importance_data.append({
                    'feature': feat_name,
                    'importance': imp,
                    'fold': i + 1
                })
        
        return pd.DataFrame(importance_data)


def get_catboost_configs():
    """获取不同的CatBoost配置"""
    configs = {
        'catboost_v0': {
            'name': 'CatBoost_V0',
            'catboost_params': {
                'learning_rate': 0.08,
                'depth': 7,
                'l2_leaf_reg': 3.0,
                'bagging_temperature': 1.0,
                'random_strength': 1.0,
                'one_hot_max_size': 2,
                'leaf_estimation_method': 'Newton',
                'bootstrap_type': 'Bayesian',
                'sampling_frequency': 'PerTree',
            },
            'num_boost_round': 3000,
            'early_stopping_rounds': 200
        },
        
        'catboost_v1': {
            'name': 'CatBoost_V1',
            'catboost_params': {
                'learning_rate': 0.05,
                'depth': 8,
                'l2_leaf_reg': 5.0,
                'bagging_temperature': 1.0,
                'random_strength': 1.0,
                'one_hot_max_size': 2,
                'leaf_estimation_method': 'Newton',
                'bootstrap_type': 'Bayesian',
                'sampling_frequency': 'PerTree',
            },
            'num_boost_round': 5000,
            'early_stopping_rounds': 300
        },
        
        'catboost_v2': {
            'name': 'CatBoost_V2',
            'catboost_params': {
                'learning_rate': 0.03,
                'depth': 9,
                'l2_leaf_reg': 8.0,
                'bagging_temperature': 1.0,
                'random_strength': 1.0,
                'one_hot_max_size': 4,
                'leaf_estimation_method': 'Newton',
                'bootstrap_type': 'Bayesian',
                'sampling_frequency': 'PerTree',
            },
            'num_boost_round': 8000,
            'early_stopping_rounds': 400
        }
    }
    return configs


def train_catboost_model(config_name, cache_dir, output_dir, **kwargs):
    """训练CatBoost模型的便捷函数"""
    from .base_model import load_processed_cache, get_default_config
    
    # 获取配置
    configs = get_catboost_configs()
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    config = get_default_config()
    config.update(configs[config_name])
    config.update(kwargs)
    config['cache_dir'] = cache_dir
    config['output_dir'] = output_dir
    
    # 加载数据
    train_data, test_data, features, cat_features, meta = load_processed_cache(cache_dir)
    config['cat_features'] = cat_features
    
    # 创建模型
    model = CatBoostModel(config)
    
    # 训练
    train_ids = train_data[config['id_col']].values
    test_ids = test_data[config['id_col']].values
    
    cv_results = model.train_cv(
        train_data=train_data,
        features=features,
        cat_features=cat_features,
        target_col=config['target'],
        n_folds=config['n_folds'],
        seed=config['random_state'],
        save_models=config['save_models'],
        model_dir=Path(output_dir) / f"{config['name']}_{model.cv_auc:.6f}"
    )
    
    # 预测测试集
    test_pred = model.predict_test(test_data, features)
    
    # 保存结果
    if config['save_predictions']:
        from .base_model import save_predictions
        save_predictions(
            oof_pred=cv_results['oof_predictions'],
            test_pred=test_pred,
            train_ids=train_ids,
            test_ids=test_ids,
            target_col=train_data[config['target']].values,
            output_dir=Path(output_dir) / f"{config['name']}_{model.cv_auc:.6f}",
            model_name=config['name']
        )
    
    # 保存特征重要性
    importance_df = model.get_feature_importance()
    if importance_df is not None:
        importance_path = Path(output_dir) / f"{config['name']}_{model.cv_auc:.6f}" / "feature_importance.csv"
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(importance_path, index=False)
        print(f"[Feature Importance] Saved to {importance_path}")
    
    return model, cv_results


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="catboost_v0", 
                       choices=list(get_catboost_configs().keys()))
    parser.add_argument("--cache_dir", type=str, default="data/processed_v1")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--num_boost_round", type=int, default=None)
    
    args = parser.parse_args()
    
    # 训练模型
    model, results = train_catboost_model(
        config_name=args.config,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        num_boost_round=args.num_boost_round
    )
    
    print(f"Training completed! CV AUC: {results['cv_auc']:.6f}")
