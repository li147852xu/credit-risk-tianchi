#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Linear Models Implementation
===========================
线性模型（Logistic Regression, Linear SVM等）的实现
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from .base_model import BaseModel


class LinearModel(BaseModel):
    """线性模型基类"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.scalers = []  # 存储每个fold的scaler
        self.model_type = config.get('model_type', 'logistic_regression')
        
    def _get_model(self):
        """获取具体的线性模型"""
        if self.model_type == 'logistic_regression':
            return LogisticRegression(
                C=self.config.get('C', 1.0),
                penalty=self.config.get('penalty', 'l2'),
                solver=self.config.get('solver', 'lbfgs'),
                max_iter=self.config.get('max_iter', 1000),
                random_state=self.config.get('random_state', 2025),
                n_jobs=os.cpu_count() or 4
            )
        elif self.model_type == 'linear_svm':
            return LinearSVC(
                C=self.config.get('C', 1.0),
                penalty=self.config.get('penalty', 'l2'),
                loss=self.config.get('loss', 'squared_hinge'),
                max_iter=self.config.get('max_iter', 1000),
                random_state=self.config.get('random_state', 2025)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train_single_fold(self, X_train, y_train, X_val, y_val, fold_idx):
        """训练单个fold的线性模型"""
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 存储scaler
        self.scalers.append(scaler)
        
        # 创建和训练模型
        model = self._get_model()
        model.fit(X_train_scaled, y_train)
        
        return model
    
    def predict_single_fold(self, model, X):
        """使用单个线性模型进行预测"""
        # 使用对应的scaler
        scaler = self.scalers[len(self.models) - 1]  # 当前模型对应的scaler
        X_scaled = scaler.transform(X)
        
        if self.model_type == 'logistic_regression':
            return model.predict_proba(X_scaled)[:, 1]
        elif self.model_type == 'linear_svm':
            # LinearSVC没有predict_proba，使用decision_function
            decision_scores = model.decision_function(X_scaled)
            # 将decision scores转换为概率（简单sigmoid）
            return 1 / (1 + np.exp(-decision_scores))
    
    def _save_model_implementation(self, model, path):
        """保存线性模型"""
        import joblib
        joblib.dump(model, str(path))
    
    def get_feature_importance(self):
        """获取特征重要性（系数）"""
        if not self.models:
            return None
            
        importance_data = []
        for i, model in enumerate(self.models):
            if hasattr(model, 'coef_'):
                coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                # 获取特征名称（如果有的话）
                feature_names = getattr(model, 'feature_names_in_', None)
                if feature_names is None:
                    feature_names = [f'feature_{j}' for j in range(len(coef))]
                
                for feat_name, coef_val in zip(feature_names, coef):
                    importance_data.append({
                        'feature': feat_name,
                        'importance': abs(coef_val),  # 使用绝对值
                        'coefficient': coef_val,
                        'fold': i + 1
                    })
        
        return pd.DataFrame(importance_data)


def get_linear_configs():
    """获取不同的线性模型配置"""
    configs = {
        'logistic_regression': {
            'name': 'LogisticRegression',
            'model_type': 'logistic_regression',
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000
        },
        
        'logistic_regression_l1': {
            'name': 'LogisticRegression_L1',
            'model_type': 'logistic_regression',
            'C': 0.1,
            'penalty': 'l1',
            'solver': 'liblinear',
            'max_iter': 1000
        },
        
        'linear_svm': {
            'name': 'LinearSVM',
            'model_type': 'linear_svm',
            'C': 1.0,
            'penalty': 'l2',
            'loss': 'squared_hinge',
            'max_iter': 1000
        },
        
        'linear_svm_l1': {
            'name': 'LinearSVM_L1',
            'model_type': 'linear_svm',
            'C': 0.1,
            'penalty': 'l1',
            'loss': 'squared_hinge',
            'max_iter': 1000
        }
    }
    return configs


def train_linear_model(config_name, cache_dir, output_dir, **kwargs):
    """训练线性模型的便捷函数"""
    from .base_model import load_processed_cache, get_default_config
    
    # 获取配置
    configs = get_linear_configs()
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
    model = LinearModel(config)
    
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
    parser.add_argument("--config", type=str, default="logistic_regression", 
                       choices=list(get_linear_configs().keys()))
    parser.add_argument("--cache_dir", type=str, default="data/processed_v1")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--C", type=float, default=None)
    
    args = parser.parse_args()
    
    # 训练模型
    model, results = train_linear_model(
        config_name=args.config,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        C=args.C
    )
    
    print(f"Training completed! CV AUC: {results['cv_auc']:.6f}")
