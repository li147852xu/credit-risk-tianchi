#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for models
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from models import (
    LightGBMModel, XGBoostModel, CatBoostModel, LinearModel,
    get_lightgbm_configs, get_xgboost_configs, get_catboost_configs, get_linear_configs
)


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(20)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['id'] = range(len(df))
    
    return df


class TestLightGBMModel:
    """Test LightGBM model"""
    
    def test_lightgbm_configs(self):
        """Test LightGBM configurations"""
        configs = get_lightgbm_configs()
        assert isinstance(configs, dict)
        assert 'lightgbm_v0' in configs
        assert 'lightgbm_v1' in configs
        assert 'lightgbm_v2' in configs
    
    def test_lightgbm_initialization(self):
        """Test LightGBM model initialization"""
        config = {'n_folds': 3, 'random_state': 42}
        model = LightGBMModel(config)
        assert isinstance(model, LightGBMModel)
        assert model.config == config
    
    def test_lightgbm_training(self, sample_data):
        """Test LightGBM training"""
        config = {
            'n_folds': 3,
            'random_state': 42,
            'num_boost_round': 10,
            'early_stopping_rounds': 5,
            'target': 'target',
            'id_col': 'id'
        }
        model = LightGBMModel(config)
        
        features = [f'feature_{i}' for i in range(20)]
        cat_features = []
        
        cv_results = model.train_cv(
            train_data=sample_data,
            features=features,
            cat_features=cat_features,
            target_col='target',
            n_folds=3,
            seed=42
        )
        
        assert 'cv_auc' in cv_results
        assert cv_results['cv_auc'] > 0.5  # Should be better than random


class TestXGBoostModel:
    """Test XGBoost model"""
    
    def test_xgboost_configs(self):
        """Test XGBoost configurations"""
        configs = get_xgboost_configs()
        assert isinstance(configs, dict)
        assert 'xgboost_v0' in configs
        assert 'xgboost_v1' in configs
        assert 'xgboost_v2' in configs
    
    def test_xgboost_initialization(self):
        """Test XGBoost model initialization"""
        config = {'n_folds': 3, 'random_state': 42}
        model = XGBoostModel(config)
        assert isinstance(model, XGBoostModel)
        assert model.config == config


class TestCatBoostModel:
    """Test CatBoost model"""
    
    def test_catboost_configs(self):
        """Test CatBoost configurations"""
        configs = get_catboost_configs()
        assert isinstance(configs, dict)
        assert 'catboost_v0' in configs
        assert 'catboost_v1' in configs
        assert 'catboost_v2' in configs
    
    def test_catboost_initialization(self):
        """Test CatBoost model initialization"""
        config = {'n_folds': 3, 'random_state': 42}
        model = CatBoostModel(config)
        assert isinstance(model, CatBoostModel)
        assert model.config == config


class TestLinearModel:
    """Test Linear model"""
    
    def test_linear_configs(self):
        """Test Linear model configurations"""
        configs = get_linear_configs()
        assert isinstance(configs, dict)
        assert 'logistic_regression' in configs
        assert 'linear_svm' in configs
    
    def test_linear_initialization(self):
        """Test Linear model initialization"""
        config = {'n_folds': 3, 'random_state': 42, 'model_type': 'logistic_regression'}
        model = LinearModel(config)
        assert isinstance(model, LinearModel)
        assert model.config == config
    
    def test_linear_training(self, sample_data):
        """Test Linear model training"""
        config = {
            'n_folds': 3,
            'random_state': 42,
            'model_type': 'logistic_regression',
            'target': 'target',
            'id_col': 'id'
        }
        model = LinearModel(config)
        
        features = [f'feature_{i}' for i in range(20)]
        cat_features = []
        
        cv_results = model.train_cv(
            train_data=sample_data,
            features=features,
            cat_features=cat_features,
            target_col='target',
            n_folds=3,
            seed=42
        )
        
        assert 'cv_auc' in cv_results
        assert cv_results['cv_auc'] > 0.5  # Should be better than random


def test_model_consistency():
    """Test that all models have consistent interfaces"""
    base_config = {'n_folds': 3, 'random_state': 42}
    
    models = [
        LightGBMModel(base_config),
        XGBoostModel(base_config),
        CatBoostModel(base_config),
        LinearModel({**base_config, 'model_type': 'logistic_regression'})
    ]
    
    for model in models:
        assert hasattr(model, 'config')
        assert hasattr(model, 'train_cv')
        assert hasattr(model, 'predict_test')
        assert hasattr(model, 'cv_auc')


if __name__ == "__main__":
    pytest.main([__file__])
