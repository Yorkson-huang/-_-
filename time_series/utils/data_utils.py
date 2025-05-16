import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns  # 导入 seaborn
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO):
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 文件处理程序
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # 控制台处理程序
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理程序
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_data(file_path):
    """
    加载北京AQI数据集
    """
    logger = logging.getLogger('data_utils')
    logger.info(f"正在加载数据集: {file_path}")
    try:
        df = pd.read_excel(file_path)
        logger.info(f"数据集加载成功，形状: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"加载数据集出错: {str(e)}")
        raise

def create_sequences(data, seq_length):
    """
    创建用于时间序列预测的序列
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def prepare_data(df, seq_length=7, test_size=0.2, val_size=0.1):
    """
    准备用于时间序列预测的数据
    """
    logger = logging.getLogger('data_utils')
    
    # 检查列名并提取AQI数据
    try:
        if 'AQI' in df.columns:
            data = df['AQI'].values
        else:
            # 假设第二列是AQI
            data = df.iloc[:, 1].values
        
        logger.info(f"提取的AQI数据形状: {data.shape}")
    except Exception as e:
        logger.error(f"提取AQI数据出错: {str(e)}")
        raise
    
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # 创建序列
    X, y = create_sequences(data_normalized, seq_length)
    logger.info(f"创建的序列形状: X={X.shape}, y={y.shape}")
    
    # 划分数据集
    train_size = int(len(X) * (1 - test_size - val_size))
    val_size = int(len(X) * val_size)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    logger.info(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"验证集形状: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"测试集形状: X={X_test.shape}, y={y_test.shape}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler,
        'seq_length': seq_length
    }

def inverse_transform(scaler, data):
    """
    将归一化的数据转换回原始范围
    """
    if len(data.shape) == 1:
        return scaler.inverse_transform(data.reshape(-1, 1)).flatten()
    else:
        return scaler.inverse_transform(data)

def plot_results(actual, predicted, title, save_path=None):
    """
    Plot comparison between predicted and actual values
    """
    sns.set_theme(style="whitegrid", palette="muted")

    plt.figure(figsize=(15, 7))
    plt.plot(actual, label='Actual', color=sns.color_palette("muted")[0], linewidth=1.5)
    plt.plot(predicted, label='Predicted', color=sns.color_palette("muted")[1], linestyle='--', linewidth=1.5)
    plt.title(title, fontsize=16)
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('AQI', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_results_to_excel(results, file_path):
    """
    将模型评估结果保存为Excel文件
    """
    df = pd.DataFrame(results)
    df.to_excel(file_path, index=False)
    print(f"结果已保存至: {file_path}") 