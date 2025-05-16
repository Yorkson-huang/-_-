import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from time_series.utils.evaluation import time_model_execution

class CNNLSTMModel(nn.Module):
    """
    结合CNN和LSTM的时间序列预测模型
    """
    def __init__(self, input_channels=1, seq_length=7, lstm_hidden_size=64, lstm_num_layers=1, output_size=1):
        super(CNNLSTMModel, self).__init__()
        
        # CNN部分
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # CNN输出序列长度
        self.cnn_output_length = seq_length // 2 + (seq_length % 2)
        
        # LSTM部分
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm = nn.LSTM(
            input_size=16,  # CNN的输出通道数
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        
        # 全连接输出层
        self.fc = nn.Linear(lstm_hidden_size, output_size)
    
    def forward(self, x):
        # x形状: [batch_size, input_channels, seq_length]
        
        # CNN特征提取
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 为LSTM准备输入: [batch_size, seq_length, channels]
        x = x.permute(0, 2, 1)
        
        # 初始化LSTM隐藏状态
        h0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        
        # LSTM处理
        out, _ = self.lstm(x, (h0, c0))
        
        # 使用最后一个时间步的输出进行预测
        out = self.fc(out[:, -1, :])
        
        return out

class CNNLSTMPredictor:
    """
    CNN+LSTM预测器，负责模型的训练、验证和预测
    """
    def __init__(self, input_channels=1, seq_length=7, lstm_hidden_size=64, lstm_num_layers=1, 
                 output_size=1, learning_rate=0.001, device=None):
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('cnn_lstm_model')
        
        # 创建模型
        self.model = CNNLSTMModel(input_channels, seq_length, lstm_hidden_size, lstm_num_layers, output_size)
        self.model.to(self.device)
        
        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.logger.info(f"CNN+LSTM模型初始化，设备: {self.device}")
        self.logger.info(f"模型结构: 输入通道={input_channels}, 序列长度={seq_length}, LSTM隐藏大小={lstm_hidden_size}")
    
    @time_model_execution
    def train(self, train_loader, val_loader, epochs=50, early_stopping_patience=10):
        """
        训练CNN+LSTM模型
        """
        self.model.train()
        best_val_loss = float('inf')
        early_stopping_counter = 0
        training_losses = []
        validation_losses = []
        
        self.logger.info(f"开始训练CNN+LSTM模型，总共{epochs}个周期")
        
        for epoch in range(epochs):
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), targets)
                
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 计算平均训练损失
            avg_train_loss = train_loss / len(train_loader)
            training_losses.append(avg_train_loss)
            
            # 验证
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.squeeze(), targets)
                    val_loss += loss.item()
            
            # 计算平均验证损失
            avg_val_loss = val_loss / len(val_loader)
            validation_losses.append(avg_val_loss)
            
            # 记录日志
            self.logger.info(f"周期 {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}")
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), 'time_series/models/best_cnn_lstm_model.pth')
                early_stopping_counter = 0
                self.logger.info(f"模型改进，已保存最佳模型")
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter >= early_stopping_patience:
                self.logger.info(f"早停触发，{early_stopping_patience}个周期内验证损失未改善")
                break
            
            # 重置为训练模式
            self.model.train()
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('time_series/models/best_cnn_lstm_model.pth'))
        self.logger.info(f"训练完成，已加载最佳模型")
        
        return training_losses, validation_losses
    
    def predict(self, test_loader):
        """
        使用训练好的模型进行预测
        """
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(targets.numpy().flatten())
        
        return np.array(predictions), np.array(actuals) 