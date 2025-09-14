# Credit Risk Prediction Project Makefile

.PHONY: help install test lint format clean setup-dev

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

setup-dev: install-dev  ## Setup development environment
	pre-commit install

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=models/ --cov-report=html --cov-report=term

lint:  ## Run linting
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:  ## Format code
	black .

type-check:  ## Run type checking
	mypy models/ --ignore-missing-imports

quality: lint type-check format  ## Run all quality checks

# Feature Engineering
fe-v1:  ## Run feature engineering v1
	python scripts/feature_engineering_v1.py --train_path data/train.csv --test_path data/testA.csv --cache_dir data/processed_v1

fe-v2:  ## Run feature engineering v2
	python scripts/feature_engineering_v2.py --train_path data/train.csv --test_path data/testA.csv --cache_dir data/processed_v2

fe-v3:  ## Run feature engineering v3
	python scripts/feature_engineering_v3.py --train_path data/train.csv --test_path data/testA.csv --out_cache_dir data/processed_v3

fe-all: fe-v1 fe-v2 fe-v3  ## Run all feature engineering versions

# Model Training
train-lightgbm:  ## Train LightGBM models
	python scripts/train_models.py --models lightgbm_v0 lightgbm_v1 lightgbm_v2 --cache_dir data/processed_v2 --output_dir outputs

train-xgboost:  ## Train XGBoost models
	python scripts/train_models.py --models xgboost_v0 xgboost_v1 xgboost_v2 --cache_dir data/processed_v2 --output_dir outputs

train-catboost:  ## Train CatBoost models
	python scripts/train_models.py --models catboost_v0 catboost_v1 catboost_v2 --cache_dir data/processed_v2 --output_dir outputs

train-linear:  ## Train Linear models
	python scripts/train_models.py --models logistic_regression linear_svm --cache_dir data/processed_v2 --output_dir outputs

train-all: train-lightgbm train-xgboost train-catboost train-linear  ## Train all models

# Model Blending
blend:  ## Blend models
	python scripts/blend.py --root_dir outputs --output_dir blend_results

# Visualization
charts:  ## Create performance charts
	python visualizations/create_charts.py

visualize: charts  ## Create visualizations (alias for charts)
	@echo "Visualization complete! Charts saved to visualizations/charts/"

# Pipeline
pipeline: fe-all train-all blend  ## Run complete pipeline

# Clean up
clean:  ## Clean up generated files
	rm -rf outputs/
	rm -rf blend_results/
	rm -rf data/processed_v*/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Docker (optional)
docker-build:  ## Build Docker image
	docker build -t credit-risk-prediction .

docker-run:  ## Run Docker container
	docker run -v $(PWD)/data:/app/data -v $(PWD)/outputs:/app/outputs credit-risk-prediction
