#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
XGBoost Model Implementation
===========================
XGBoost模型的实现，支持多种参数配置
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost模型实现"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_params = self._get_default_params()
        self.model_params.update(config.get('xgb_params', {}))
        
    def _get_default_params(self):
        """获取默认XGBoost参数"""
        return {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'predictor': 'cpu_predictor',
            'enable_categorical': True,
            'learning_rate': 0.06,
            'max_depth': 8,
            'grow_policy': 'depthwise',
            'max_bin': 256,
            'min_child_weight': 5.0,
            'gamma': 0.1,
            'lambda': 20.0,
            'alpha': 2.0,
            'subsample': 0.80,
            'colsample_bytree': 0.70,
            'colsample_bylevel': 0.70,
            'colsample_bynode': 0.70,
            'max_cat_to_onehot': 128,
            'categorical_smoothing': 10,
            'nthread': os.cpu_count() or 4,
            'verbosity': 0,
            'random_state': self.config.get('random_state', 2025)
        }
    
    def train_single_fold(self, X_train, y_train, X_val, y_val, fold_idx):
        """训练单个fold的XGBoost模型"""
        # 创建数据集
        train_dataset = xgb.DMatrix(
            X_train, 
            label=y_train,
            enable_categorical=True
        )
        
        val_dataset = xgb.DMatrix(
            X_val,
            label=y_val,
            enable_categorical=True
        )
        
        # 训练参数
        num_boost_round = self.config.get('num_boost_round', 5000)
        early_stopping_rounds = self.config.get('early_stopping_rounds', 200)
        
        # 训练模型
        model = xgb.train(
            params=self.model_params,
            dtrain=train_dataset,
            num_boost_round=num_boost_round,
            evals=[(train_dataset, 'train'), (val_dataset, 'valid')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )
        
        return model
    
    def predict_single_fold(self, model, X):
        """使用单个XGBoost模型进行预测"""
        dmat = xgb.DMatrix(X, enable_categorical=True)
        return model.predict(dmat)
    
    def _save_model_implementation(self, model, path):
        """保存XGBoost模型"""
        model.save_model(str(path))
    
    def get_feature_importance(self, importance_type='weight'):
        """获取特征重要性"""
        if not self.models:
            return None
            
        importance_data = []
        for i, model in enumerate(self.models):
            importance = model.get_score(importance_type=importance_type)
            
            for feat_name, imp in importance.items():
                importance_data.append({
                    'feature': feat_name,
                    'importance': imp,
                    'fold': i + 1
                })
        
        return pd.DataFrame(importance_data)


def get_xgboost_configs():
    """获取不同的XGBoost配置"""
    configs = {
        'xgboost_v0': {
            'name': 'XGBoost_V0',
            'xgb_params': {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'learning_rate': 0.02,
                'max_depth': 0,
                'max_leaves': 255,
                'min_child_weight': 1.0,
                'lambda': 5.0,
                'alpha': 1.0,
                'subsample': 0.85,
                'colsample_bytree': 0.9,
                'max_bin': 512,
                'enable_categorical': True,
                'nthread': os.cpu_count() or 4,
            },
            'num_boost_round': 10000,
            'early_stopping_rounds': 200
        },
        
        'xgboost_v1': {
            'name': 'XGBoost_V1',
            'xgb_params': {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'predictor': 'cpu_predictor',
                'enable_categorical': True,
                'learning_rate': 0.06,
                'max_depth': 8,
                'grow_policy': 'depthwise',
                'max_bin': 256,
                'min_child_weight': 5.0,
                'gamma': 0.1,
                'lambda': 20.0,
                'alpha': 2.0,
                'subsample': 0.80,
                'colsample_bytree': 0.70,
                'colsample_bylevel': 0.70,
                'colsample_bynode': 0.70,
                'max_cat_to_onehot': 128,
                'categorical_smoothing': 10,
            },
            'num_boost_round': 8000,
            'early_stopping_rounds': 300
        },
        
        'xgboost_v2': {
            'name': 'XGBoost_V2',
            'xgb_params': {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'predictor': 'cpu_predictor',
                'enable_categorical': True,
                'learning_rate': 0.03,
                'max_depth': 10,
                'grow_policy': 'lossguide',
                'max_leaves': 127,
                'max_bin': 512,
                'min_child_weight': 3.0,
                'gamma': 0.2,
                'lambda': 15.0,
                'alpha': 1.5,
                'subsample': 0.85,
                'colsample_bytree': 0.8,
                'max_cat_to_onehot': 256,
                'categorical_smoothing': 15,
            },
            'num_boost_round': 6000,
            'early_stopping_rounds': 250
        }
    }
    return configs


def train_xgboost_model(config_name, cache_dir, output_dir, **kwargs):
    """训练XGBoost模型的便捷函数"""
    from .base_model import load_processed_cache, get_default_config
    
    # 获取配置
    configs = get_xgboost_configs()
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
    model = XGBoostModel(config)
    
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
    parser.add_argument("--config", type=str, default="xgboost_v0", 
                       choices=list(get_xgboost_configs().keys()))
    parser.add_argument("--cache_dir", type=str, default="data/processed_v1")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--num_boost_round", type=int, default=None)
    
    args = parser.parse_args()
    
    # 训练模型
    model, results = train_xgboost_model(
        config_name=args.config,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        num_boost_round=args.num_boost_round
    )
    
    print(f"Training completed! CV AUC: {results['cv_auc']:.6f}")
