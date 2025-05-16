import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import logging
import time
from time_series.utils.evaluation import time_model_execution

class ARIMAPredictor:
    """
    ARIMA 时间序列预测模型
    """
    def __init__(self, order=(5,1,0)):
        self.order = order
        self.model = None
        self.logger = logging.getLogger('arima_model')
        self.logger.info(f"初始化ARIMA模型，阶数: {order}")

    @time_model_execution
    def train(self, train_data):
        """
        训练ARIMA模型
        Args:
            train_data (pd.Series or np.array): 训练时间序列数据 (单变量)
        """
        self.logger.info(f"开始训练ARIMA模型，数据长度: {len(train_data)}")
        # 确保 train_data 是 pd.Series
        if isinstance(train_data, np.ndarray):
            train_data = pd.Series(train_data.flatten())
        
        try:
            self.model = ARIMA(train_data, order=self.order)
            self.model_fit = self.model.fit()
            self.logger.info("ARIMA模型训练完成")
            self.logger.info(self.model_fit.summary())
        except Exception as e:
            self.logger.error(f"ARIMA模型训练失败: {e}")
            # 可以选择一个更简单的模型或者默认参数
            self.logger.warning("尝试使用默认阶数 (1,1,1) 重新训练")
            try:
                self.order = (1,1,1)
                self.model = ARIMA(train_data, order=self.order)
                self.model_fit = self.model.fit()
                self.logger.info("ARIMA模型 (1,1,1) 训练完成")
                self.logger.info(self.model_fit.summary())
            except Exception as e2:
                self.logger.error(f"ARIMA模型再次训练失败: {e2}")
                self.model_fit = None # 确保出错时 model_fit 为 None
                raise
        return self.model_fit

    def predict(self, steps):
        """
        使用训练好的ARIMA模型进行预测
        Args:
            steps (int): 需要预测的步数
        Returns:
            np.array: 预测结果
        """
        if self.model_fit is None:
            self.logger.error("ARIMA模型未训练或训练失败，无法预测")
            # 返回一个与 steps 长度相同的 NaN 数组，或者根据需求返回其他
            return np.full(steps, np.nan) 
            
        self.logger.info(f"ARIMA模型预测未来 {steps} 步")
        forecast = self.model_fit.forecast(steps=steps)
        return forecast.values # Prophet 返回 DataFrame，ARIMA 返回 Series 