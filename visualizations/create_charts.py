#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Results Visualization Script
============================
创建项目结果的各类图表可视化
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_model_results():
    """加载所有模型结果数据"""
    results = {
        'single_models': {},
        'blend_results': {}
    }
    
    # 定义模型结果数据
    single_model_data = {
        'CatBoost': {
            'C0': {'FE1': 0.7387, 'FE2': 0.7411, 'FE3': 0.7386},
            'C1': {'FE1': 0.7386, 'FE2': 0.7409, 'FE3': 0.7384}
        },
        'LightGBM': {
            'L0': {'FE1': 0.7315, 'FE2': 0.7341, 'FE3': 0.7342},
            'L1': {'FE1': 0.7332, 'FE2': 0.7359, 'FE3': 0.7362},
            'L2': {'FE1': 0.7310, 'FE2': 0.7341, 'FE3': 0.7337}
        },
        'XGBoost': {
            'X0': {'FE1': 0.7333, 'FE2': 0.7359, 'FE3': 0.7361},
            'X1': {'FE1': 0.7349, 'FE2': 0.7371, 'FE3': 0.7376},
            'X2': {'FE1': 0.7355, 'FE2': 0.7380, 'FE3': 0.7373}
        },
        'Linear': {
            'LR': {'FE1': 0.7118, 'FE2': 0.7258, 'FE3': 0.7197},
            'LS': {'FE1': 0.7120, 'FE2': 0.7246, 'FE3': 0.7195}
        }
    }
    
    # 定义融合结果数据
    blend_data = {
        'FE1': {
            'Weight Optimization': 0.7418,
            'Greedy Selection': 0.7418,
            'Stacking LR': 0.7417,
            'Stacking Ridge': 0.7415,
            'Simple Mean': 0.7392
        },
        'FE2': {
            'Weight Optimization': 0.7418,
            'Greedy Selection': 0.7418,
            'Stacking LR': 0.7417,
            'Stacking Ridge': 0.7414,
            'Simple Mean': 0.7401
        },
        'FE3': {
            'Weight Optimization': 0.7414,
            'Stacking LR': 0.7414,
            'Greedy Selection': 0.7414,
            'Stacking Ridge': 0.7407,
            'Simple Mean': 0.7392
        },
        'FE1+2+3': {
            'Weight Optimization': 0.7418,
            'Greedy Selection': 0.7418,
            'Stacking LR': 0.7417,
            'Stacking Ridge': 0.7415,
            'Simple Mean': 0.7392
        },
        'FE2+3': {
            'Weight Optimization': 0.7414,
            'Stacking LR': 0.7414,
            'Greedy Selection': 0.7414,
            'Stacking Ridge': 0.7407,
            'Simple Mean': 0.7392
        }
    }
    
    return single_model_data, blend_data

def create_model_comparison_chart(single_model_data, output_dir):
    """创建模型对比图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison Across Feature Engineering Versions', fontsize=16, fontweight='bold')
    
    model_types = list(single_model_data.keys())
    fe_versions = ['FE1', 'FE2', 'FE3']
    
    for idx, model_type in enumerate(model_types):
        ax = axes[idx // 2, idx % 2]
        
        # 准备数据
        data_for_plot = []
        for model_name, fe_results in single_model_data[model_type].items():
            for fe_version in fe_versions:
                data_for_plot.append({
                    'Model': f"{model_type} {model_name}",
                    'FE Version': fe_version,
                    'AUC': fe_results[fe_version]
                })
        
        df = pd.DataFrame(data_for_plot)
        
        # 绘制分组柱状图
        sns.barplot(data=df, x='Model', y='AUC', hue='FE Version', ax=ax)
        ax.set_title(f'{model_type} Performance', fontweight='bold')
        ax.set_ylabel('AUC Score')
        ax.set_xlabel('Model Variants')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Feature Engineering')
        
        # 添加数值标签
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', rotation=90, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_fe_improvement_chart(single_model_data, output_dir):
    """创建特征工程改进图表"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 计算每个模型在FE1到FE2和FE2到FE3的改进
    improvements = []
    
    for model_type, models in single_model_data.items():
        for model_name, fe_results in models.items():
            fe1_to_fe2 = fe_results['FE2'] - fe_results['FE1']
            fe2_to_fe3 = fe_results['FE3'] - fe_results['FE2']
            
            improvements.append({
                'Model': f"{model_type} {model_name}",
                'FE1→FE2': fe1_to_fe2,
                'FE2→FE3': fe2_to_fe3
            })
    
    df = pd.DataFrame(improvements)
    
    # 绘制分组柱状图
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['FE1→FE2'], width, label='FE1→FE2', alpha=0.8)
    bars2 = ax.bar(x + width/2, df['FE2→FE3'], width, label='FE2→FE3', alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('AUC Improvement')
    ax.set_title('Feature Engineering Improvement by Model', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.4f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fe_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_blend_comparison_chart(blend_data, output_dir):
    """创建融合策略对比图表"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 准备数据
    data_for_plot = []
    for fe_version, strategies in blend_data.items():
        for strategy, auc in strategies.items():
            data_for_plot.append({
                'FE Version': fe_version,
                'Strategy': strategy,
                'AUC': auc
            })
    
    df = pd.DataFrame(data_for_plot)
    
    # 绘制分组柱状图
    sns.barplot(data=df, x='FE Version', y='AUC', hue='Strategy', ax=ax)
    ax.set_title('Blending Strategy Comparison Across Feature Engineering Versions', fontweight='bold')
    ax.set_ylabel('AUC Score')
    ax.set_xlabel('Feature Engineering Version')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', rotation=90, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'blend_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_best_results_chart(single_model_data, blend_data, output_dir):
    """创建最佳结果对比图表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 最佳单模型结果
    best_single = []
    for model_type, models in single_model_data.items():
        for model_name, fe_results in models.items():
            best_auc = max(fe_results.values())
            best_single.append({
                'Model': f"{model_type} {model_name}",
                'Best AUC': best_auc
            })
    
    df_single = pd.DataFrame(best_single).sort_values('Best AUC', ascending=False)
    
    bars1 = ax1.bar(range(len(df_single)), df_single['Best AUC'], alpha=0.7)
    ax1.set_title('Best Single Model Performance', fontweight='bold')
    ax1.set_ylabel('AUC Score')
    ax1.set_xlabel('Models')
    ax1.set_xticks(range(len(df_single)))
    ax1.set_xticklabels(df_single['Model'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 最佳融合结果
    best_blend = []
    for fe_version, strategies in blend_data.items():
        best_auc = max(strategies.values())
        best_blend.append({
            'FE Version': fe_version,
            'Best AUC': best_auc
        })
    
    df_blend = pd.DataFrame(best_blend).sort_values('Best AUC', ascending=False)
    
    bars2 = ax2.bar(range(len(df_blend)), df_blend['Best AUC'], alpha=0.7, color='orange')
    ax2.set_title('Best Blending Performance', fontweight='bold')
    ax2.set_ylabel('AUC Score')
    ax2.set_xlabel('Feature Engineering Version')
    ax2.set_xticks(range(len(df_blend)))
    ax2.set_xticklabels(df_blend['FE Version'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_heatmap_chart(single_model_data, output_dir):
    """创建热力图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 准备热力图数据
    heatmap_data = []
    model_labels = []
    
    for model_type, models in single_model_data.items():
        for model_name, fe_results in models.items():
            model_labels.append(f"{model_type} {model_name}")
            heatmap_data.append([fe_results['FE1'], fe_results['FE2'], fe_results['FE3']])
    
    heatmap_array = np.array(heatmap_data)
    
    # 创建热力图
    im = ax.imshow(heatmap_array, cmap='YlOrRd', aspect='auto')
    
    # 设置标签
    ax.set_xticks(range(3))
    ax.set_xticklabels(['FE1', 'FE2', 'FE3'])
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels)
    
    # 添加数值标签
    for i in range(len(model_labels)):
        for j in range(3):
            text = ax.text(j, i, f'{heatmap_array[i, j]:.4f}',
                         ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Model Performance Heatmap Across Feature Engineering Versions', fontweight='bold')
    ax.set_xlabel('Feature Engineering Version')
    ax.set_ylabel('Models')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('AUC Score', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_stats_chart(single_model_data, blend_data, output_dir):
    """创建统计摘要图表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Statistics Summary', fontsize=16, fontweight='bold')
    
    # 1. 模型类型平均性能
    model_type_avg = {}
    for model_type, models in single_model_data.items():
        all_aucs = []
        for fe_results in models.values():
            all_aucs.extend(fe_results.values())
        model_type_avg[model_type] = np.mean(all_aucs)
    
    ax1.bar(model_type_avg.keys(), model_type_avg.values(), alpha=0.7)
    ax1.set_title('Average Performance by Model Type')
    ax1.set_ylabel('Average AUC')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. FE版本平均性能
    fe_avg = {}
    for model_type, models in single_model_data.items():
        for model_name, fe_results in models.items():
            for fe_version, auc in fe_results.items():
                if fe_version not in fe_avg:
                    fe_avg[fe_version] = []
                fe_avg[fe_version].append(auc)
    
    fe_avg = {fe: np.mean(aucs) for fe, aucs in fe_avg.items()}
    ax2.bar(fe_avg.keys(), fe_avg.values(), alpha=0.7, color='green')
    ax2.set_title('Average Performance by FE Version')
    ax2.set_ylabel('Average AUC')
    
    # 3. 融合策略平均性能
    strategy_avg = {}
    for fe_version, strategies in blend_data.items():
        for strategy, auc in strategies.items():
            if strategy not in strategy_avg:
                strategy_avg[strategy] = []
            strategy_avg[strategy].append(auc)
    
    strategy_avg = {strategy: np.mean(aucs) for strategy, aucs in strategy_avg.items()}
    ax3.bar(strategy_avg.keys(), strategy_avg.values(), alpha=0.7, color='purple')
    ax3.set_title('Average Performance by Blending Strategy')
    ax3.set_ylabel('Average AUC')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 性能分布
    all_single_aucs = []
    for model_type, models in single_model_data.items():
        for fe_results in models.values():
            all_single_aucs.extend(fe_results.values())
    
    all_blend_aucs = []
    for strategies in blend_data.values():
        all_blend_aucs.extend(strategies.values())
    
    ax4.hist(all_single_aucs, bins=15, alpha=0.7, label='Single Models', color='blue')
    ax4.hist(all_blend_aucs, bins=15, alpha=0.7, label='Blended Models', color='red')
    ax4.set_title('AUC Score Distribution')
    ax4.set_xlabel('AUC Score')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    print("开始创建结果可视化图表...")
    
    # 创建输出目录
    output_dir = Path('visualizations/charts')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    single_model_data, blend_data = load_model_results()
    
    print("创建模型对比图表...")
    create_model_comparison_chart(single_model_data, output_dir)
    
    print("创建特征工程改进图表...")
    create_fe_improvement_chart(single_model_data, output_dir)
    
    print("创建融合策略对比图表...")
    create_blend_comparison_chart(blend_data, output_dir)
    
    print("创建最佳结果图表...")
    create_best_results_chart(single_model_data, blend_data, output_dir)
    
    print("创建性能热力图...")
    create_heatmap_chart(single_model_data, output_dir)
    
    print("创建统计摘要图表...")
    create_summary_stats_chart(single_model_data, blend_data, output_dir)
    
    print(f"所有图表已保存到: {output_dir}")
    print("图表创建完成！")

if __name__ == "__main__":
    main()
