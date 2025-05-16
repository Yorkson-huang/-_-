import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from time_series.utils.evaluation import time_model_execution

class LSTMModel(nn.Module):
    """
    基于LSTM的时间序列预测模型
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 检查输入维度
        if len(x.shape) == 2:
            # 如果是2D输入，添加批次维度
            x = x.unsqueeze(0)
        
        # 初始化隐藏状态
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 获取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        
        # 如果输入是2D，则移除批次维度
        if len(x.shape) == 3 and x.size(0) == 1:
            out = out.squeeze(0)
        
        return out

class LSTMPredictor:
    """
    LSTM预测器，负责模型的训练、验证和预测
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, 
                 learning_rate=0.001, device=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('lstm_model')
        
        # 创建模型
        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        self.model.to(self.device)
        
        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.logger.info(f"LSTM模型初始化，设备: {self.device}")
        self.logger.info(f"模型结构: 输入大小={input_size}, 隐藏大小={hidden_size}, 层数={num_layers}")
    
    @time_model_execution
    def train(self, train_loader, val_loader, epochs=50, early_stopping_patience=10):
        """
        训练LSTM模型
        """
        self.model.train()
        best_val_loss = float('inf')
        early_stopping_counter = 0
        training_losses = []
        validation_losses = []
        
        self.logger.info(f"开始训练LSTM模型，总共{epochs}个周期")
        
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
                torch.save(self.model.state_dict(), 'time_series/models/best_lstm_model.pth')
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
        self.model.load_state_dict(torch.load('time_series/models/best_lstm_model.pth'))
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