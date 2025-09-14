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
    """创建模型对比图表 - 突出差异"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Model Performance Comparison (AUC Improvement Analysis)', fontsize=18, fontweight='bold')
    
    model_types = list(single_model_data.keys())
    fe_versions = ['FE1', 'FE2', 'FE3']
    
    # 计算全局基准（最小AUC）
    all_aucs = []
    for model_type, models in single_model_data.items():
        for fe_results in models.values():
            all_aucs.extend(fe_results.values())
    baseline_auc = min(all_aucs)
    
    for idx, model_type in enumerate(model_types):
        ax = axes[idx // 2, idx % 2]
        
        # 准备数据 - 计算相对于基准的提升
        data_for_plot = []
        for model_name, fe_results in single_model_data[model_type].items():
            for fe_version in fe_versions:
                improvement = (fe_results[fe_version] - baseline_auc) * 10000  # 放大10000倍
                data_for_plot.append({
                    'Model': f"{model_name}",
                    'FE Version': fe_version,
                    'Improvement': improvement,
                    'Original AUC': fe_results[fe_version]
                })
        
        df = pd.DataFrame(data_for_plot)
        
        # 绘制分组柱状图
        sns.barplot(data=df, x='Model', y='Improvement', hue='FE Version', ax=ax)
        ax.set_title(f'{model_type} Performance (vs Baseline +{baseline_auc:.4f})', fontweight='bold', fontsize=12)
        ax.set_ylabel('AUC Improvement (×10,000)')
        ax.set_xlabel('Model Variants')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Feature Engineering', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加原始AUC值作为标签
        for i, container in enumerate(ax.containers):
            for j, bar in enumerate(container):
                height = bar.get_height()
                original_auc = df.iloc[j * len(fe_versions) + i]['Original AUC']
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{original_auc:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_fe_improvement_chart(single_model_data, output_dir):
    """创建特征工程改进图表 - 突出改进幅度"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 计算每个模型在FE1到FE2和FE2到FE3的改进
    improvements = []
    
    for model_type, models in single_model_data.items():
        for model_name, fe_results in models.items():
            fe1_to_fe2 = (fe_results['FE2'] - fe_results['FE1']) * 10000  # 放大10000倍
            fe2_to_fe3 = (fe_results['FE3'] - fe_results['FE2']) * 10000  # 放大10000倍
            fe1_to_fe3 = (fe_results['FE3'] - fe_results['FE1']) * 10000  # 放大10000倍
            
            improvements.append({
                'Model': f"{model_type} {model_name}",
                'FE1→FE2': fe1_to_fe2,
                'FE2→FE3': fe2_to_fe3,
                'FE1→FE3': fe1_to_fe3,
                'FE1_AUC': fe_results['FE1'],
                'FE2_AUC': fe_results['FE2'],
                'FE3_AUC': fe_results['FE3']
            })
    
    df = pd.DataFrame(improvements)
    
    # 绘制分组柱状图
    x = np.arange(len(df))
    width = 0.25
    
    bars1 = ax.bar(x - width, df['FE1→FE2'], width, label='FE1→FE2', alpha=0.8, color='#2E8B57')
    bars2 = ax.bar(x, df['FE2→FE3'], width, label='FE2→FE3', alpha=0.8, color='#4169E1')
    bars3 = ax.bar(x + width, df['FE1→FE3'], width, label='FE1→FE3 (Total)', alpha=0.8, color='#DC143C')
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('AUC Improvement (×10,000)', fontsize=12)
    ax.set_title('Feature Engineering Improvement Analysis (Amplified Scale)', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签 - 显示原始改进值
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        original_improvement = df.iloc[i]['FE1→FE2'] / 10000
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{original_improvement:+.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        original_improvement = df.iloc[i]['FE2→FE3'] / 10000
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{original_improvement:+.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        original_improvement = df.iloc[i]['FE1→FE3'] / 10000
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{original_improvement:+.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 添加零线
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
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
    """创建最佳结果对比图表 - 突出差异"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 最佳单模型结果 - 使用相对改进
    best_single = []
    baseline_auc = 0.71  # 设置一个合理的基准
    
    for model_type, models in single_model_data.items():
        for model_name, fe_results in models.items():
            best_auc = max(fe_results.values())
            improvement = (best_auc - baseline_auc) * 10000  # 放大10000倍
            best_single.append({
                'Model': f"{model_type} {model_name}",
                'Best AUC': best_auc,
                'Improvement': improvement
            })
    
    df_single = pd.DataFrame(best_single).sort_values('Best AUC', ascending=False)
    
    # 使用改进值绘制，但标签显示原始AUC
    bars1 = ax1.bar(range(len(df_single)), df_single['Improvement'], alpha=0.7, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(df_single))))
    ax1.set_title('Best Single Model Performance (vs Baseline 0.7100)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('AUC Improvement (×10,000)')
    ax1.set_xlabel('Models')
    ax1.set_xticks(range(len(df_single)))
    ax1.set_xticklabels(df_single['Model'], rotation=45, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 添加原始AUC值作为标签
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        original_auc = df_single.iloc[i]['Best AUC']
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{original_auc:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 最佳融合结果
    best_blend = []
    for fe_version, strategies in blend_data.items():
        best_auc = max(strategies.values())
        improvement = (best_auc - baseline_auc) * 10000  # 放大10000倍
        best_blend.append({
            'FE Version': fe_version,
            'Best AUC': best_auc,
            'Improvement': improvement
        })
    
    df_blend = pd.DataFrame(best_blend).sort_values('Best AUC', ascending=False)
    
    bars2 = ax2.bar(range(len(df_blend)), df_blend['Improvement'], alpha=0.7, 
                   color=plt.cm.plasma(np.linspace(0, 1, len(df_blend))))
    ax2.set_title('Best Blending Performance (vs Baseline 0.7100)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('AUC Improvement (×10,000)')
    ax2.set_xlabel('Feature Engineering Version')
    ax2.set_xticks(range(len(df_blend)))
    ax2.set_xticklabels(df_blend['FE Version'], rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 添加原始AUC值作为标签
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        original_auc = df_blend.iloc[i]['Best AUC']
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{original_auc:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_heatmap_chart(single_model_data, output_dir):
    """创建热力图 - 显示改进幅度"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 准备热力图数据 - 原始AUC
    heatmap_data_original = []
    model_labels = []
    
    for model_type, models in single_model_data.items():
        for model_name, fe_results in models.items():
            model_labels.append(f"{model_type} {model_name}")
            heatmap_data_original.append([fe_results['FE1'], fe_results['FE2'], fe_results['FE3']])
    
    heatmap_array_original = np.array(heatmap_data_original)
    
    # 左图：原始AUC热力图
    im1 = ax1.imshow(heatmap_array_original, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(['FE1', 'FE2', 'FE3'], fontsize=12)
    ax1.set_yticks(range(len(model_labels)))
    ax1.set_yticklabels(model_labels, fontsize=10)
    ax1.set_title('Original AUC Scores', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Feature Engineering Version')
    ax1.set_ylabel('Models')
    
    # 添加数值标签
    for i in range(len(model_labels)):
        for j in range(3):
            text = ax1.text(j, i, f'{heatmap_array_original[i, j]:.4f}',
                           ha="center", va="center", color="white", fontweight='bold', fontsize=9)
    
    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('AUC Score', rotation=270, labelpad=20)
    
    # 右图：改进幅度热力图
    baseline_auc = min(heatmap_array_original.flatten())
    improvement_data = (heatmap_array_original - baseline_auc) * 10000  # 放大10000倍
    
    im2 = ax2.imshow(improvement_data, cmap='RdYlGn', aspect='auto')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['FE1', 'FE2', 'FE3'], fontsize=12)
    ax2.set_yticks(range(len(model_labels)))
    ax2.set_yticklabels(model_labels, fontsize=10)
    ax2.set_title(f'Improvement vs Baseline ({baseline_auc:.4f})', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Feature Engineering Version')
    ax2.set_ylabel('Models')
    
    # 添加数值标签 - 显示原始改进值
    for i in range(len(model_labels)):
        for j in range(3):
            original_improvement = (heatmap_array_original[i, j] - baseline_auc)
            text = ax2.text(j, i, f'{original_improvement:+.4f}',
                           ha="center", va="center", color="black", fontweight='bold', fontsize=9)
    
    # 添加颜色条
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Improvement (×10,000)', rotation=270, labelpad=20)
    
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
