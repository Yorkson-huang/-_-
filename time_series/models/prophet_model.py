import pandas as pd
from prophet import Prophet
import logging
import time
from time_series.utils.evaluation import time_model_execution

class ProphetPredictor:
    """
    Prophet 时间序列预测模型
    """
    def __init__(self, **kwargs):
        self.model = Prophet(**kwargs)
        self.logger = logging.getLogger('prophet_model')
        self.logger.info(f"初始化Prophet模型，参数: {kwargs}")

    @time_model_execution
    def train(self, train_df):
        """
        训练Prophet模型
        Args:
            train_df (pd.DataFrame): 训练数据，必须包含 'ds' (日期) 和 'y' (值) 两列
        """
        self.logger.info("开始训练Prophet模型")
        if not all(col in train_df.columns for col in ['ds', 'y']):
            self.logger.error("Prophet训练数据必须包含 'ds' 和 'y' 列")
            raise ValueError("Prophet训练数据必须包含 'ds' 和 'y' 列")
        
        try:
            self.model.fit(train_df)
            self.logger.info("Prophet模型训练完成")
        except Exception as e:
            self.logger.error(f"Prophet模型训练失败: {e}")
            raise
        return self.model

    def predict(self, future_df):
        """
        使用训练好的Prophet模型进行预测
        Args:
            future_df (pd.DataFrame): 需要预测的未来日期，必须包含 'ds' 列
        Returns:
            pd.DataFrame: 预测结果，包含 'ds', 'yhat', 'yhat_lower', 'yhat_upper' 等列
        """
        self.logger.info("Prophet模型进行预测")
        if 'ds' not in future_df.columns:
            self.logger.error("Prophet预测数据必须包含 'ds' 列")
            raise ValueError("Prophet预测数据必须包含 'ds' 列")
            
        forecast = self.model.predict(future_df)
        return forecast 