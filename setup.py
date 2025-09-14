#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for Credit Risk Prediction Project
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="credit-risk-prediction",
    version="1.0.0",
    author="Credit Risk Team",
    author_email="",
    description="Credit Risk Prediction for Tianchi Competition",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/li147852xu/credit-risk-tianchi",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "gpu": [
            "lightgbm[gpu]",
            "xgboost[gpu]",
        ]
    },
    entry_points={
        "console_scripts": [
            "train-models=scripts.train_models:main",
            "blend-models=scripts.blend:main",
            "fe-v1=scripts.feature_engineering_v1:main",
            "fe-v2=scripts.feature_engineering_v2:main", 
            "fe-v3=scripts.feature_engineering_v3:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
