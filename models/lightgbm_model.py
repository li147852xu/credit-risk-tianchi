#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LightGBM Model Implementation
=============================
LightGBM模型的实现，支持多种参数配置
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from .base_model import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM模型实现"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_params = self._get_default_params()
        self.model_params.update(config.get('lgb_params', {}))
        
    def _get_default_params(self):
        """获取默认LightGBM参数"""
        return {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 127,
            'max_depth': -1,
            'max_bin': 255,
            'min_data_in_leaf': 100,
            'min_sum_hessian_in_leaf': 1.0,
            'lambda_l1': 0.0,
            'lambda_l2': 5.0,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'force_col_wise': True,
            'verbosity': -1,
            'num_threads': os.cpu_count() or 4,
            'deterministic': True,
            'seed': self.config.get('random_state', 2025)
        }
    
    def train_single_fold(self, X_train, y_train, X_val, y_val, fold_idx):
        """训练单个fold的LightGBM模型"""
        # 创建数据集
        train_dataset = lgb.Dataset(
            X_train, 
            label=y_train,
            categorical_feature=self.config.get('cat_features', []),
            free_raw_data=True
        )
        
        val_dataset = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=self.config.get('cat_features', []),
            free_raw_data=True,
            reference=train_dataset
        )
        
        # 训练参数
        num_boost_round = self.config.get('num_boost_round', 3000)
        early_stopping_rounds = self.config.get('early_stopping_rounds', 200)
        
        # 回调函数
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0)
        ]
        
        # 训练模型
        model = lgb.train(
            params=self.model_params,
            train_set=train_dataset,
            num_boost_round=num_boost_round,
            valid_sets=[train_dataset, val_dataset],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
        
        return model
    
    def predict_single_fold(self, model, X):
        """使用单个LightGBM模型进行预测"""
        return model.predict(X, num_iteration=model.best_iteration)
    
    def _save_model_implementation(self, model, path):
        """保存LightGBM模型"""
        model.save_model(str(path))
    
    def get_feature_importance(self, importance_type='gain'):
        """获取特征重要性"""
        if not self.models:
            return None
            
        importance_data = []
        for i, model in enumerate(self.models):
            importance = model.feature_importance(importance_type=importance_type)
            feature_names = model.feature_name()
            
            for feat_name, imp in zip(feature_names, importance):
                importance_data.append({
                    'feature': feat_name,
                    'importance': imp,
                    'fold': i + 1
                })
        
        return pd.DataFrame(importance_data)


def get_lightgbm_configs():
    """获取不同的LightGBM配置"""
    configs = {
        'lightgbm_v0': {
            'name': 'LightGBM_V0',
            'lgb_params': {
                'learning_rate': 0.10,
                'num_leaves': 63,
                'max_depth': 9,
                'min_data_in_leaf': 200,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'lambda_l1': 0.0,
                'lambda_l2': 0.0,
                'max_bin': 255,
                'bin_construct_sample_cnt': 200000,
            },
            'num_boost_round': 3000,
            'early_stopping_rounds': 100
        },
        
        'lightgbm_v1': {
            'name': 'LightGBM_V1',
            'lgb_params': {
                'learning_rate': 0.01,
                'num_leaves': 255,
                'max_depth': -1,
                'max_bin': 511,
                'min_data_in_leaf': 50,
                'min_sum_hessian_in_leaf': 1.0,
                'lambda_l1': 1.0,
                'lambda_l2': 5.0,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.85,
                'max_cat_threshold': 1024,
                'cat_smooth': 10.0,
                'cat_l2': 10.0,
                'bin_construct_sample_cnt': 500000,
            },
            'num_boost_round': 8000,
            'early_stopping_rounds': 400
        },
        
        'lightgbm_v2': {
            'name': 'LightGBM_V2',
            'lgb_params': {
                'learning_rate': 0.02,
                'num_leaves': 191,
                'max_depth': 10,
                'max_bin': 255,
                'min_data_in_leaf': 80,
                'lambda_l1': 0.5,
                'lambda_l2': 3.0,
                'feature_fraction': 0.85,
                'bagging_fraction': 0.8,
                'bin_construct_sample_cnt': 300000,
            },
            'num_boost_round': 5000,
            'early_stopping_rounds': 300
        }
    }
    return configs


def train_lightgbm_model(config_name, cache_dir, output_dir, **kwargs):
    """训练LightGBM模型的便捷函数"""
    from .base_model import load_processed_cache, get_default_config
    
    # 获取配置
    configs = get_lightgbm_configs()
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
    model = LightGBMModel(config)
    
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
    parser.add_argument("--config", type=str, default="lightgbm_v0", 
                       choices=list(get_lightgbm_configs().keys()))
    parser.add_argument("--cache_dir", type=str, default="data/processed_v1")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--num_boost_round", type=int, default=None)
    
    args = parser.parse_args()
    
    # 训练模型
    model, results = train_lightgbm_model(
        config_name=args.config,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        num_boost_round=args.num_boost_round
    )
    
    print(f"Training completed! CV AUC: {results['cv_auc']:.6f}")
