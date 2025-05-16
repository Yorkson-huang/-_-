import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import logging

class TimeSeriesDataProcessor:
    """
    时间序列数据预处理器，负责数据的预处理和加载
    """
    def __init__(self, data_dict, batch_size=32):
        self.X_train = data_dict['X_train']
        self.y_train = data_dict['y_train']
        self.X_val = data_dict['X_val']
        self.y_val = data_dict['y_val']
        self.X_test = data_dict['X_test']
        self.y_test = data_dict['y_test']
        self.scaler = data_dict['scaler']
        self.seq_length = data_dict['seq_length']
        self.batch_size = batch_size
        self.logger = logging.getLogger('data_processor')
        
        self.logger.info(f"数据处理器初始化，序列长度: {self.seq_length}, 批次大小: {batch_size}")
    
    def prepare_pytorch_data(self):
        """
        将数据转换为PyTorch可用的格式
        """
        # 重塑数据为LSTM所需的形状 (batch_size, sequence_length, features)
        X_train_reshaped = self.reshape_for_lstm(self.X_train)
        X_val_reshaped = self.reshape_for_lstm(self.X_val)
        X_test_reshaped = self.reshape_for_lstm(self.X_test)
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_reshaped)
        y_train_tensor = torch.FloatTensor(self.y_train)
        X_val_tensor = torch.FloatTensor(X_val_reshaped)
        y_val_tensor = torch.FloatTensor(self.y_val)
        X_test_tensor = torch.FloatTensor(X_test_reshaped)
        y_test_tensor = torch.FloatTensor(self.y_test)
        
        # 创建TensorDataset
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        self.logger.info(f"PyTorch数据准备完成")
        self.logger.info(f"训练批次数: {len(train_loader)}")
        self.logger.info(f"验证批次数: {len(val_loader)}")
        self.logger.info(f"测试批次数: {len(test_loader)}")
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'X_train_tensor': X_train_tensor,
            'y_train_tensor': y_train_tensor,
            'X_val_tensor': X_val_tensor,
            'y_val_tensor': y_val_tensor,
            'X_test_tensor': X_test_tensor,
            'y_test_tensor': y_test_tensor
        }
    
    def prepare_sklearn_data(self):
        """
        准备用于sklearn模型的数据
        """
        # 对于基于树的模型，我们可能需要重塑数据
        X_train_reshaped = self.X_train.reshape(self.X_train.shape[0], -1)
        X_val_reshaped = self.X_val.reshape(self.X_val.shape[0], -1)
        X_test_reshaped = self.X_test.reshape(self.X_test.shape[0], -1)
        
        self.logger.info(f"Sklearn数据准备完成")
        self.logger.info(f"重塑后的训练数据形状: {X_train_reshaped.shape}")
        self.logger.info(f"重塑后的验证数据形状: {X_val_reshaped.shape}")
        self.logger.info(f"重塑后的测试数据形状: {X_test_reshaped.shape}")
        
        return {
            'X_train': X_train_reshaped,
            'y_train': self.y_train,
            'X_val': X_val_reshaped,
            'y_val': self.y_val,
            'X_test': X_test_reshaped,
            'y_test': self.y_test
        }
    
    def get_cnn_data_shape(self):
        """
        获取CNN模型所需的数据形状
        """
        # CNN需要额外的通道维度
        return {
            'input_shape': (1, self.seq_length),  # (channels, sequence_length)
            'output_shape': 1
        }
    
    def get_lstm_data_shape(self):
        """
        获取LSTM模型所需的数据形状
        """
        return {
            'input_shape': (self.seq_length, 1),  # (sequence_length, features)
            'output_shape': 1
        }
    
    def reshape_for_cnn(self, data):
        """
        重塑数据为CNN所需的形状
        """
        # 添加通道维度，形状为 (batch_size, channels, sequence_length)
        return data.reshape(data.shape[0], 1, data.shape[1])
    
    def reshape_for_lstm(self, data):
        """
        重塑数据为LSTM所需的形状
        """
        # 添加特征维度
        return data.reshape(data.shape[0], data.shape[1], 1) 