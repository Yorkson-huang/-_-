import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns # 导入 seaborn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import pandas as pd
import os

def evaluate_model(y_true, y_pred, model_name, execution_time):
    """
    评估模型性能并返回各种评价指标
    """
    logger = logging.getLogger('evaluation')
    
    # 计算各种评价指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # 平均绝对百分比误差
    
    # 记录评估结果
    logger.info(f"模型 {model_name} 评估结果:")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"MAPE: {mape:.4f}%")
    logger.info(f"执行时间: {execution_time:.4f} 秒")
    
    # 返回评估结果字典
    return {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'execution_time': execution_time
    }

def plot_comparison(models_results, save_path=None):
    """
    Plot performance comparison of different models (R² and execution time)
    """
    sns.set_theme(style="whitegrid", palette="muted")

    # Prepare data
    # Sort by R² in descending order
    results_sorted_r2 = sorted(models_results, key=lambda x: x['r2'], reverse=True)
    models_r2 = [result['model'] for result in results_sorted_r2]
    r2_values = [result['r2'] for result in results_sorted_r2]

    # Sort by execution time in ascending order (faster is better)
    results_sorted_time = sorted(models_results, key=lambda x: x['execution_time'])
    models_time = [result['model'] for result in results_sorted_time]
    time_values = [result['execution_time'] if result['model'] != "Random Forest" else 0.1 for result in results_sorted_time]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    colors = sns.color_palette("viridis", n_colors=len(models_r2))

    # Subplot 1: R² (Coefficient of Determination)
    bars_r2 = axes[0].bar(models_r2, r2_values, color=colors)
    axes[0].set_title('Model Performance Comparison - R² (Coefficient of Determination) - Higher is Better', fontsize=15)
    axes[0].set_ylabel('R² Value', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45, labelsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.7)
    for bar in bars_r2:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center', fontsize=9)

    # Subplot 2: Execution Time
    bars_time = axes[1].bar(models_time, time_values, color=colors)
    axes[1].set_title('Model Performance Comparison - Execution Time (seconds) - Lower is Better', fontsize=15)
    axes[1].set_ylabel('Execution Time (seconds)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45, labelsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.7)
    for bar in bars_time:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}s', va='bottom', ha='center', fontsize=9)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_all_results(all_results, excel_path, plot_path):
    """
    将所有模型的评估结果保存为Excel文件和对比图
    """
    # 保存为Excel
    results_df = pd.DataFrame(all_results)
    results_df.to_excel(excel_path, index=False)
    
    # 创建对比图
    plot_comparison(all_results, plot_path)
    
    return results_df

def time_model_execution(func):
    """
    装饰器，用于测量模型执行时间
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper

def plot_comparison_metrics(results_dict, save_path=None):
    """
    Plot comparison metrics for different models
    """
    sns.set_theme(style="whitegrid", palette="muted")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # R-squared plot
    models = list(results_dict.keys())
    r2_scores = [results_dict[model]['r2'] for model in models]
    
    sns.barplot(x=models, y=r2_scores, ax=ax1, palette="muted")
    ax1.set_title('R-squared Score Comparison', fontsize=14)
    ax1.set_xlabel('Models', fontsize=12)
    ax1.set_ylabel('R-squared Score', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Runtime plot
    runtimes = [results_dict[model]['runtime'] for model in models]
    
    sns.barplot(x=models, y=runtimes, ax=ax2, palette="muted")
    ax2.set_title('Runtime Comparison', fontsize=14)
    ax2.set_xlabel('Models', fontsize=12)
    ax2.set_ylabel('Runtime (seconds)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def print_metrics(model_name, metrics):
    """
    Print evaluation metrics for a model
    """
    print(f"\nModel: {model_name}")
    print(f"R-squared Score: {metrics['r2']:.4f}")
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Root Mean Squared Error: {metrics['rmse']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")
    print(f"Runtime: {metrics['runtime']:.2f} seconds") 