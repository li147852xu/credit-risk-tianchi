#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Model Class for Credit Risk Prediction
==========================================
统一的模型基类，包含通用的训练、验证和预测逻辑
"""

import os
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


class BaseModel(ABC):
    """模型基类"""
    
    def __init__(self, config: dict):
        self.config = config
        self.models = []  # 存储fold模型
        self.cv_auc = 0.0
        self.oof_predictions = None
        self.test_predictions = None
        self.feature_importance = []
        
    @abstractmethod
    def train_single_fold(self, X_train, y_train, X_val, y_val, fold_idx):
        """训练单个fold的模型"""
        pass
        
    @abstractmethod
    def predict_single_fold(self, model, X):
        """使用单个模型进行预测"""
        pass
        
    def train_cv(self, train_data, features, cat_features, target_col, 
                 n_folds=5, seed=2025, save_models=False, model_dir=None):
        """交叉验证训练"""
        print(f"[{self.__class__.__name__}] Starting CV training...")
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        n_samples = len(train_data)
        
        self.oof_predictions = np.zeros(n_samples)
        self.test_predictions = np.zeros(self.config.get('test_size', 200000))
        
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_data[features], train_data[target_col])):
            print(f"\n[Fold {fold_idx + 1}/{n_folds}]")
            
            X_train = train_data.iloc[train_idx][features]
            y_train = train_data.iloc[train_idx][target_col]
            X_val = train_data.iloc[val_idx][features]
            y_val = train_data.iloc[val_idx][target_col]
            
            # 训练模型
            model = self.train_single_fold(X_train, y_train, X_val, y_val, fold_idx)
            self.models.append(model)
            
            # 验证集预测
            val_pred = self.predict_single_fold(model, X_val)
            self.oof_predictions[val_idx] = val_pred
            
            # 验证集AUC
            val_auc = roc_auc_score(y_val, val_pred)
            print(f"[Fold {fold_idx + 1}] Val AUC: {val_auc:.6f}")
            
            fold_results.append({
                'fold': fold_idx + 1,
                'val_auc': val_auc,
                'n_train': len(X_train),
                'n_val': len(X_val)
            })
            
            # 保存模型
            if save_models and model_dir:
                self._save_model(model, fold_idx, model_dir)
                
        # 计算整体CV AUC
        self.cv_auc = roc_auc_score(train_data[target_col], self.oof_predictions)
        print(f"\n[CV] Overall AUC: {self.cv_auc:.6f}")
        
        return {
            'cv_auc': self.cv_auc,
            'fold_results': fold_results,
            'oof_predictions': self.oof_predictions
        }
    
    def predict_test(self, test_data, features):
        """预测测试集"""
        if not self.models:
            raise ValueError("No trained models found. Please run train_cv first.")
            
        print(f"[{self.__class__.__name__}] Predicting test set...")
        
        test_preds = np.zeros(len(test_data))
        
        for i, model in enumerate(self.models):
            fold_pred = self.predict_single_fold(model, test_data[features])
            test_preds += fold_pred / len(self.models)
            
        self.test_predictions = test_preds
        return test_preds
    
    def _save_model(self, model, fold_idx, model_dir):
        """保存模型"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{self.__class__.__name__.lower()}_fold_{fold_idx + 1}.model"
        self._save_model_implementation(model, model_path)
        print(f"[Fold {fold_idx + 1}] Model saved to {model_path}")
    
    @abstractmethod
    def _save_model_implementation(self, model, path):
        """具体的模型保存实现"""
        pass
    
    def save_metadata(self, model_dir, additional_meta=None):
        """保存训练元数据"""
        if not model_dir:
            return
            
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'model_type': self.__class__.__name__,
            'cv_auc': self.cv_auc,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'n_folds': len(self.models)
        }
        
        if additional_meta:
            metadata.update(additional_meta)
            
        meta_path = model_dir / 'metadata.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        print(f"[Metadata] Saved to {meta_path}")


def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_processed_cache(cache_dir, target_col="isDefault", id_col="id"):
    """加载处理后的缓存数据"""
    cache_dir = Path(cache_dir)
    
    # 检查缓存是否存在
    train_path = cache_dir / "train_fe.csv"
    test_path = cache_dir / "test_fe.csv"
    meta_path = cache_dir / "meta.json"
    
    if not all([train_path.exists(), test_path.exists(), meta_path.exists()]):
        raise FileNotFoundError(f"Cache files not found in {cache_dir}")
    
    # 加载数据
    train_data = pd.read_csv(train_path, low_memory=False)
    test_data = pd.read_csv(test_path, low_memory=False)
    
    # 加载元数据
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    features = meta['features']
    cat_features = meta.get('cat_features', [])
    
    print(f"[Cache] Loaded train: {train_data.shape}, test: {test_data.shape}")
    print(f"[Cache] Features: {len(features)}, Cat features: {len(cat_features)}")
    
    return train_data, test_data, features, cat_features, meta


def save_predictions(oof_pred, test_pred, train_ids, test_ids, target_col, 
                    output_dir, model_name, id_col="id"):
    """保存预测结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存OOF预测
    oof_df = pd.DataFrame({
        id_col: train_ids,
        'oof_pred': oof_pred,
        target_col: target_col  # 这里需要传入真实的target值
    })
    oof_path = output_dir / f"oof_{model_name}.csv"
    oof_df.to_csv(oof_path, index=False)
    
    # 保存测试集预测
    test_df = pd.DataFrame({
        id_col: test_ids,
        'isDefault': np.clip(test_pred, 0, 1)
    })
    test_path = output_dir / f"submission_{model_name}.csv"
    test_df.to_csv(test_path, index=False)
    
    print(f"[Save] OOF predictions: {oof_path}")
    print(f"[Save] Test predictions: {test_path}")


def get_default_config():
    """获取默认配置"""
    return {
        'n_folds': 5,
        'random_state': 2025,
        'save_models': True,
        'save_predictions': True,
        'cache_dir': 'data/processed_v1',
        'output_dir': 'outputs',
        'target': 'isDefault',
        'id_col': 'id'
    }
