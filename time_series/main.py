import os
import sys
import numpy as np
import pandas as pd
import torch
import logging
import time
import matplotlib.pyplot as plt
from datetime import datetime
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from utils.data_utils import load_data, prepare_data, inverse_transform, plot_results, save_results_to_excel, setup_logger
from utils.evaluation import evaluate_model, save_all_results
from preprocessing.data_processor import TimeSeriesDataProcessor
from models.lstm_model import LSTMPredictor
from models.cnn_model import CNNPredictor
from models.cnn_lstm_model import CNNLSTMPredictor
from models.sklearn_models import SklearnTimeSeriesModels
from models.arima_model import ARIMAPredictor
from models.prophet_model import ProphetPredictor
from models.transformer_model import TransformerPredictor
from models.cnn_lstm_attention_model import CNNLSTMAttentionPredictor

def run_lstm_model(data_processor, epochs=50):
    """运行LSTM模型"""
    logger.info("=" * 50)
    logger.info("开始LSTM模型训练和预测")
    
    # 准备数据
    data_dict = data_processor.prepare_pytorch_data()
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    test_loader = data_dict['test_loader']
    
    # 获取LSTM所需的数据形状
    shape_dict = data_processor.get_lstm_data_shape()
    
    # 初始化模型
    predictor = LSTMPredictor(
        input_size=shape_dict['input_shape'][1],  # 特征数
        hidden_size=64,
        num_layers=2,
        output_size=shape_dict['output_shape']
    )
    
    # 训练模型
    model_result, execution_time = predictor.train(train_loader, val_loader, epochs=epochs)
    
    # 预测
    predictions, actuals = predictor.predict(test_loader)
    
    # 反归一化预测结果
    predictions_original = inverse_transform(data_processor.scaler, predictions)
    actuals_original = inverse_transform(data_processor.scaler, actuals)
    
    # 评估模型
    evaluation_result = evaluate_model(actuals_original, predictions_original, "LSTM", execution_time)
    
    # 绘制结果
    plot_path = os.path.join("time_series/results", "lstm_prediction.png")
    plot_results(actuals_original, predictions_original, "LSTM Model Prediction Results", plot_path)
    
    logger.info(f"LSTM模型完成，评估结果: RMSE={evaluation_result['rmse']:.4f}, MAE={evaluation_result['mae']:.4f}")
    
    return evaluation_result

def run_cnn_model(data_processor, epochs=50):
    """运行CNN模型"""
    logger.info("=" * 50)
    logger.info("开始CNN模型训练和预测")
    
    # 准备数据
    pytorch_data = data_processor.prepare_pytorch_data()
    
    # 获取CNN所需的数据形状
    shape_dict = data_processor.get_cnn_data_shape()
    
    # 重塑训练、验证和测试数据
    X_train_reshaped = data_processor.reshape_for_cnn(pytorch_data['X_train_tensor'].numpy())
    X_val_reshaped = data_processor.reshape_for_cnn(pytorch_data['X_val_tensor'].numpy())
    X_test_reshaped = data_processor.reshape_for_cnn(pytorch_data['X_test_tensor'].numpy())
    
    # 创建新的数据加载器
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train_reshaped),
        pytorch_data['y_train_tensor']
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val_reshaped),
        pytorch_data['y_val_tensor']
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test_reshaped),
        pytorch_data['y_test_tensor']
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=data_processor.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=data_processor.batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=data_processor.batch_size
    )
    
    # 初始化模型
    predictor = CNNPredictor(
        input_channels=shape_dict['input_shape'][0],  # 输入通道数
        seq_length=shape_dict['input_shape'][1],      # 序列长度
        output_size=shape_dict['output_shape']
    )
    
    # 训练模型
    model_result, execution_time = predictor.train(train_loader, val_loader, epochs=epochs)
    
    # 预测
    predictions, actuals = predictor.predict(test_loader)
    
    # 反归一化预测结果
    predictions_original = inverse_transform(data_processor.scaler, predictions)
    actuals_original = inverse_transform(data_processor.scaler, actuals)
    
    # 评估模型
    evaluation_result = evaluate_model(actuals_original, predictions_original, "CNN", execution_time)
    
    # 绘制结果
    plot_path = os.path.join("time_series/results", "cnn_prediction.png")
    plot_results(actuals_original, predictions_original, "CNN Model Prediction Results", plot_path)
    
    logger.info(f"CNN模型完成，评估结果: RMSE={evaluation_result['rmse']:.4f}, MAE={evaluation_result['mae']:.4f}")
    
    return evaluation_result

def run_cnn_lstm_model(data_processor, epochs=50):
    """运行CNN+LSTM模型"""
    logger.info("=" * 50)
    logger.info("开始CNN+LSTM模型训练和预测")
    
    # 准备数据
    pytorch_data = data_processor.prepare_pytorch_data()
    
    # 获取CNN所需的数据形状
    shape_dict = data_processor.get_cnn_data_shape()
    
    # 重塑训练、验证和测试数据
    X_train_reshaped = data_processor.reshape_for_cnn(pytorch_data['X_train_tensor'].numpy())
    X_val_reshaped = data_processor.reshape_for_cnn(pytorch_data['X_val_tensor'].numpy())
    X_test_reshaped = data_processor.reshape_for_cnn(pytorch_data['X_test_tensor'].numpy())
    
    # 创建新的数据加载器
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train_reshaped),
        pytorch_data['y_train_tensor']
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val_reshaped),
        pytorch_data['y_val_tensor']
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test_reshaped),
        pytorch_data['y_test_tensor']
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=data_processor.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=data_processor.batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=data_processor.batch_size
    )
    
    # 初始化模型
    predictor = CNNLSTMPredictor(
        input_channels=shape_dict['input_shape'][0],  # 输入通道数
        seq_length=shape_dict['input_shape'][1],      # 序列长度
        lstm_hidden_size=64,
        lstm_num_layers=1,
        output_size=shape_dict['output_shape']
    )
    
    # 训练模型
    model_result, execution_time = predictor.train(train_loader, val_loader, epochs=epochs)
    
    # 预测
    predictions, actuals = predictor.predict(test_loader)
    
    # 反归一化预测结果
    predictions_original = inverse_transform(data_processor.scaler, predictions)
    actuals_original = inverse_transform(data_processor.scaler, actuals)
    
    # 评估模型
    evaluation_result = evaluate_model(actuals_original, predictions_original, "CNN+LSTM", execution_time)
    
    # 绘制结果
    plot_path = os.path.join("time_series/results", "cnn_lstm_prediction.png")
    plot_results(actuals_original, predictions_original, "CNN+LSTM Model Prediction Results", plot_path)
    
    logger.info(f"CNN+LSTM模型完成，评估结果: RMSE={evaluation_result['rmse']:.4f}, MAE={evaluation_result['mae']:.4f}")
    
    return evaluation_result

def run_sklearn_models(data_processor):
    """运行基于Sklearn的模型"""
    logger.info("=" * 50)
    logger.info("开始Sklearn模型训练和预测")
    
    # 准备数据
    sklearn_data = data_processor.prepare_sklearn_data()
    X_train = sklearn_data['X_train']
    y_train = sklearn_data['y_train']
    X_val = sklearn_data['X_val']
    y_val = sklearn_data['y_val']
    X_test = sklearn_data['X_test']
    y_test = sklearn_data['y_test']
    
    # 初始化模型集合
    models = SklearnTimeSeriesModels()
    evaluation_results = []
    y_test_original = inverse_transform(data_processor.scaler, y_test) # 只需反归一化一次真实值
    
    # 训练线性回归模型
    logger.info("-" * 30)
    logger.info("训练线性回归模型")
    lr_model, lr_time = models.train_linear_regression(X_train, y_train, X_val, y_val)
    lr_preds = models.predict('linear_regression', X_test)
    lr_preds_original = inverse_transform(data_processor.scaler, lr_preds)
    lr_eval = evaluate_model(y_test_original, lr_preds_original, "线性回归", lr_time)
    evaluation_results.append(lr_eval)
    plot_path = os.path.join("time_series/results", "linear_regression_prediction.png")
    plot_results(y_test_original, lr_preds_original, "Linear Regression Model Prediction Results", plot_path)
    
    # 训练决策树模型
    logger.info("-" * 30)
    logger.info("训练决策树模型")
    dt_model, dt_time = models.train_decision_tree(X_train, y_train, X_val, y_val)
    dt_preds = models.predict('decision_tree', X_test)
    dt_preds_original = inverse_transform(data_processor.scaler, dt_preds)
    dt_eval = evaluate_model(y_test_original, dt_preds_original, "决策树", dt_time)
    evaluation_results.append(dt_eval)
    plot_path = os.path.join("time_series/results", "decision_tree_prediction.png")
    plot_results(y_test_original, dt_preds_original, "Decision Tree Model Prediction Results", plot_path)
    
    # 训练随机森林模型
    logger.info("-" * 30)
    logger.info("训练随机森林模型")
    rf_model, rf_time = models.train_random_forest(X_train, y_train, X_val, y_val)
    rf_preds = models.predict('random_forest', X_test)
    rf_preds_original = inverse_transform(data_processor.scaler, rf_preds)
    rf_eval = evaluate_model(y_test_original, rf_preds_original, "随机森林", rf_time)
    evaluation_results.append(rf_eval)
    plot_path = os.path.join("time_series/results", "random_forest_prediction.png")
    plot_results(y_test_original, rf_preds_original, "Random Forest Model Prediction Results", plot_path)
    
    # 训练XGBoost模型
    logger.info("-" * 30)
    logger.info("训练XGBoost模型")
    xgb_model, xgb_time = models.train_xgboost(X_train, y_train, X_val, y_val)
    xgb_preds = models.predict('xgboost', X_test)
    xgb_preds_original = inverse_transform(data_processor.scaler, xgb_preds)
    xgb_eval = evaluate_model(y_test_original, xgb_preds_original, "XGBoost", xgb_time)
    evaluation_results.append(xgb_eval)
    plot_path = os.path.join("time_series/results", "xgboost_prediction.png")
    plot_results(y_test_original, xgb_preds_original, "XGBoost Model Prediction Results", plot_path)

    # 训练SVM模型
    logger.info("-" * 30)
    logger.info("训练SVM模型")
    svm_model, svm_time = models.train_svm(X_train, y_train, X_val, y_val)
    svm_preds = models.predict('svm', X_test)
    svm_preds_original = inverse_transform(data_processor.scaler, svm_preds)
    svm_eval = evaluate_model(y_test_original, svm_preds_original, "SVM", svm_time)
    evaluation_results.append(svm_eval)
    plot_path = os.path.join("time_series/results", "svm_prediction.png")
    plot_results(y_test_original, svm_preds_original, "SVM Model Prediction Results", plot_path)

    # 训练梯度提升树模型
    logger.info("-" * 30)
    logger.info("训练梯度提升树模型")
    gb_model, gb_time = models.train_gradient_boosting(X_train, y_train, X_val, y_val)
    gb_preds = models.predict('gradient_boosting', X_test)
    gb_preds_original = inverse_transform(data_processor.scaler, gb_preds)
    gb_eval = evaluate_model(y_test_original, gb_preds_original, "梯度提升树", gb_time)
    evaluation_results.append(gb_eval)
    plot_path = os.path.join("time_series/results", "gradient_boosting_prediction.png")
    plot_results(y_test_original, gb_preds_original, "Gradient Boosting Model Prediction Results", plot_path)

    # 训练LightGBM模型
    logger.info("-" * 30)
    logger.info("训练LightGBM模型")
    lgbm_model, lgbm_time = models.train_lightgbm(X_train, y_train, X_val, y_val)
    lgbm_preds = models.predict('lightgbm', X_test)
    lgbm_preds_original = inverse_transform(data_processor.scaler, lgbm_preds)
    lgbm_eval = evaluate_model(y_test_original, lgbm_preds_original, "LightGBM", lgbm_time)
    evaluation_results.append(lgbm_eval)
    plot_path = os.path.join("time_series/results", "lightgbm_prediction.png")
    plot_results(y_test_original, lgbm_preds_original, "LightGBM Model Prediction Results", plot_path)

    # 训练CatBoost模型
    logger.info("-" * 30)
    logger.info("训练CatBoost模型")
    cb_model, cb_time = models.train_catboost(X_train, y_train, X_val, y_val)
    cb_preds = models.predict('catboost', X_test)
    cb_preds_original = inverse_transform(data_processor.scaler, cb_preds)
    cb_eval = evaluate_model(y_test_original, cb_preds_original, "CatBoost", cb_time)
    evaluation_results.append(cb_eval)
    plot_path = os.path.join("time_series/results", "catboost_prediction.png")
    plot_results(y_test_original, cb_preds_original, "CatBoost Model Prediction Results", plot_path)
    
    logger.info("所有Sklearn模型训练完成")
    
    return evaluation_results

def run_arima_model(data_processor, original_data_df):
    """运行ARIMA模型"""
    logger.info("=" * 50)
    logger.info("开始ARIMA模型训练和预测")

    # 准备数据
    # ARIMA 通常使用单个时间序列进行训练。
    # 我们将使用验证集之前的所有数据（训练集+部分历史数据，如果seq_length>1）的 'y' 值作为训练数据
    # 但为了简化并与其他模型对齐，我们使用 y_train (归一化)
    # 注意：ARIMA的输入是单一序列，所以我们用 y_train (目标值)
    # 但实际上，ARIMA 通常基于历史值来预测， prepare_data 已经将数据切分了
    # 我们这里使用 y_train + y_val 作为训练集，在 y_test 上预测，以利用更多数据
    # 或者，更简单地，直接使用原始序列的训练部分。

    # 获取原始数据中训练集和验证集的部分，以进行ARIMA训练
    # data_dict = data_processor.data_dict # 获取 data_processor 内部的 data_dict
    # y_train_arima = data_dict['y_train'] # 这是归一化后的
    # y_val_arima = data_dict['y_val']
    # train_arima_series = np.concatenate((y_train_arima, y_val_arima)).flatten()
    
    # 为了保持与其他模型在测试集上的一致性，我们用原始数据中对应训练+验证集的部分，但进行归一化
    # 原始数据 df['AQI']
    aqi_series = data_processor.data_dict['scaler'].transform(original_data_df['AQI'].values.reshape(-1, 1)).flatten()
    train_val_split_idx = len(data_processor.data_dict['y_train']) + len(data_processor.data_dict['y_val'])
    train_arima_series = aqi_series[:train_val_split_idx]

    y_test = data_processor.data_dict['y_test'].flatten() # 归一化后的真实测试值
    
    # 初始化ARIMA模型
    # order (p,d,q) - p: AR项, d: 差分阶数, q: MA项. 这些参数通常需要通过ACF/PACF图或网格搜索确定
    # 这里使用一个常用的默认值，实际应用中需要调优
    arima_predictor = ARIMAPredictor(order=(5,1,0)) 
    
    # 训练模型
    model_fit, execution_time = arima_predictor.train(train_arima_series)
    
    if model_fit is None: # 检查模型是否成功训练
        logger.error("ARIMA模型训练失败，跳过评估。")
        return {
            'model': "ARIMA", 'rmse': np.nan, 'mae': np.nan, 
            'r2': np.nan, 'mape': np.nan, 'execution_time': execution_time
        }

    # 预测
    n_test_steps = len(y_test)
    predictions_normalized = arima_predictor.predict(steps=n_test_steps)

    if np.isnan(predictions_normalized).all(): # 检查预测是否都是 NaN
        logger.error("ARIMA模型预测失败，全为NaN，跳过评估。")
        return {
            'model': "ARIMA", 'rmse': np.nan, 'mae': np.nan, 
            'r2': np.nan, 'mape': np.nan, 'execution_time': execution_time
        }
        
    # 反归一化预测结果
    # 注意：ARIMA预测的是差分后的序列，如果d>0。statsmodels的forecast会自动处理还原。
    # 我们训练时用的是归一化数据，所以预测结果也是归一化的尺度。
    predictions_original = inverse_transform(data_processor.scaler, predictions_normalized)
    actuals_original = inverse_transform(data_processor.scaler, y_test) # y_test 已经是正确的测试集真实值

    # 评估模型
    evaluation_result = evaluate_model(actuals_original, predictions_original, "ARIMA", execution_time)
    
    # 绘制结果
    plot_path = os.path.join("time_series/results", "arima_prediction.png")
    plot_results(actuals_original, predictions_original, "ARIMA Model Prediction Results", plot_path)
    
    logger.info(f"ARIMA模型完成，评估结果: RMSE={evaluation_result['rmse']:.4f}, MAE={evaluation_result['mae']:.4f}, R2={evaluation_result['r2']:.4f}")
    
    return evaluation_result

def run_prophet_model(data_processor, original_data_df):
    """运行Prophet模型"""
    logger.info("=" * 50)
    logger.info("开始Prophet模型训练和预测")

    # 准备数据
    # Prophet 需要 DataFrame，包含 'ds' (日期) 和 'y' (值) 列
    # 我们使用原始数据的日期列，和归一化后的AQI值
    
    # 假设原始DataFrame的日期列名为'日期'或第一列
    date_col_name = '日期' # 需要根据实际Excel文件的列名调整
    if date_col_name not in original_data_df.columns:
        if original_data_df.columns[0] == '日期': # 常见情况
             date_col_name = original_data_df.columns[0]
        else: # 如果没有明确的日期列，尝试使用索引作为日期替代，但这通常不适合Prophet
            logger.warning(f"Prophet模型：未找到明确的 '{date_col_name}' 列，将尝试使用索引生成日期。")
            # original_data_df['ds'] = pd.to_datetime(original_data_df.index) # 这行可能需要调整
            # 更好的做法是确保数据加载时日期列被正确解析和命名
            # 这里假设原始数据中已经有了一个可用的日期列，或者在load_data时处理了
            # 为了能继续运行，如果找不到'日期'，就用第一列
            date_col_name = original_data_df.columns[0]
            logger.info(f"Prophet模型：将使用第一列 '{date_col_name}' 作为日期列。")


    prophet_df = pd.DataFrame()
    try:
        prophet_df['ds'] = pd.to_datetime(original_data_df[date_col_name])
    except Exception as e:
        logger.error(f"Prophet: 转换日期列 '{date_col_name}' 失败: {e}. 请确保该列包含有效的日期格式。")
        # 返回失败结果
        return {
            'model': "Prophet", 'rmse': np.nan, 'mae': np.nan, 
            'r2': np.nan, 'mape': np.nan, 'execution_time': 0
        }

    # 使用归一化后的AQI值
    aqi_normalized = data_processor.data_dict['scaler'].transform(original_data_df['AQI'].values.reshape(-1, 1)).flatten()
    prophet_df['y'] = aqi_normalized

    # 划分训练集和测试集的时间点
    # 测试集长度与 data_processor 中的 y_test 保持一致
    n_test_points = len(data_processor.data_dict['y_test'])
    
    train_df_prophet = prophet_df.iloc[:-n_test_points]
    test_df_prophet = prophet_df.iloc[-n_test_points:] # 用于生成 future_df 和获取真实值

    if train_df_prophet.empty:
        logger.error("Prophet模型：训练数据为空，无法继续。")
        return {
            'model': "Prophet", 'rmse': np.nan, 'mae': np.nan, 
            'r2': np.nan, 'mape': np.nan, 'execution_time': 0
        }

    # 初始化Prophet模型
    # 可以添加节假日、自定义季节性等参数
    prophet_predictor = ProphetPredictor(daily_seasonality=True) # 示例：添加日季节性
    
    # 训练模型
    model, execution_time = prophet_predictor.train(train_df_prophet)

    if model is None: # 检查模型是否成功训练
        logger.error("Prophet模型训练失败，跳过评估。")
        return {
            'model': "Prophet", 'rmse': np.nan, 'mae': np.nan, 
            'r2': np.nan, 'mape': np.nan, 'execution_time': execution_time
        }
        
    # 创建未来日期 DataFrame 进行预测
    future_df = prophet_predictor.model.make_future_dataframe(periods=n_test_points, freq='D') # 假设数据是每日的
    
    # 预测
    forecast_result = prophet_predictor.predict(future_df)
    
    # 提取预测值 (yhat)
    # Prophet 的预测结果是从整个历史+未来时间段的，我们需要取最后 n_test_points 个点
    predictions_normalized = forecast_result['yhat'].iloc[-n_test_points:].values
    
    # 获取归一化后的真实测试值
    actuals_normalized = test_df_prophet['y'].values

    # 反归一化
    predictions_original = inverse_transform(data_processor.scaler, predictions_normalized)
    actuals_original = inverse_transform(data_processor.scaler, actuals_normalized)
    
    # 评估模型
    evaluation_result = evaluate_model(actuals_original, predictions_original, "Prophet", execution_time)
    
    # 绘制结果
    plot_path = os.path.join("time_series/results", "prophet_prediction.png")
    # Prophet 有自己的绘图工具，但为了统一，我们使用通用的 plot_results
    # prophet_predictor.model.plot(forecast_result).savefig(plot_path) # Prophet 自带绘图
    plot_results(actuals_original, predictions_original, "Prophet Model Prediction Results", plot_path)
    
    logger.info(f"Prophet模型完成，评估结果: RMSE={evaluation_result['rmse']:.4f}, MAE={evaluation_result['mae']:.4f}, R2={evaluation_result['r2']:.4f}")
    
    return evaluation_result

def run_transformer_model(data_processor, epochs=50):
    """运行Transformer模型"""
    logger.info("=" * 50)
    logger.info("开始Transformer模型训练和预测")
    
    # 准备数据 (与LSTM类似)
    data_dict = data_processor.prepare_pytorch_data()
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    test_loader = data_dict['test_loader']
    
    # 获取Transformer所需的数据形状
    # input_shape for TransformerModel is (input_size for the linear embedding layer)
    # seq_length is also a parameter for TransformerModel
    # data_processor.get_lstm_data_shape() gives [seq_len, features]
    lstm_shape_dict = data_processor.get_lstm_data_shape()
    input_size = lstm_shape_dict['input_shape'][1]  # num_features
    seq_length = lstm_shape_dict['input_shape'][0] # seq_length
    output_size = lstm_shape_dict['output_shape']
    
    # 初始化模型
    # 参数可以根据需要调整
    predictor = TransformerPredictor(
        input_size=input_size,
        d_model=64, # Embedding dimension
        nhead=4,    # Number of attention heads
        num_encoder_layers=2,
        dim_feedforward=128,
        output_size=output_size,
        seq_length=seq_length,
        dropout=0.1,
        learning_rate=0.001
    )
    
    # 训练模型
    model_result, execution_time = predictor.train(train_loader, val_loader, epochs=epochs)
    
    # 预测
    predictions, actuals = predictor.predict(test_loader)
    
    # 反归一化预测结果
    predictions_original = inverse_transform(data_processor.scaler, predictions)
    actuals_original = inverse_transform(data_processor.scaler, actuals)
    
    # 评估模型
    evaluation_result = evaluate_model(actuals_original, predictions_original, "Transformer", execution_time)
    
    # 绘制结果
    plot_path = os.path.join("time_series/results", "transformer_prediction.png")
    plot_results(actuals_original, predictions_original, "Transformer Model Prediction Results", plot_path)
    
    logger.info(f"Transformer模型完成，评估结果: RMSE={evaluation_result['rmse']:.4f}, MAE={evaluation_result['mae']:.4f}, R2={evaluation_result['r2']:.4f}")
    
    return evaluation_result

def run_cnn_lstm_attention_model(data_processor, epochs=50):
    """运行CNN-LSTM-Attention模型"""
    logger.info("=" * 50)
    logger.info("开始CNN-LSTM-Attention模型训练和预测")
    
    # 准备数据 (与CNN/CNN-LSTM类似)
    pytorch_data = data_processor.prepare_pytorch_data() # 获取原始Tensor数据
    shape_dict = data_processor.get_cnn_data_shape() # 获取CNN期望的输入形状 [channels, seq_len]

    # 重塑训练、验证和测试数据以匹配CNN的输入 [N, C, S]
    X_train_reshaped = data_processor.reshape_for_cnn(pytorch_data['X_train_tensor'].numpy())
    X_val_reshaped = data_processor.reshape_for_cnn(pytorch_data['X_val_tensor'].numpy())
    X_test_reshaped = data_processor.reshape_for_cnn(pytorch_data['X_test_tensor'].numpy())
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train_reshaped),
        pytorch_data['y_train_tensor']
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val_reshaped),
        pytorch_data['y_val_tensor']
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test_reshaped),
        pytorch_data['y_test_tensor']
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=data_processor.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=data_processor.batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=data_processor.batch_size
    )
    
    # 初始化模型
    # shape_dict['input_shape'] is [channels, seq_length]
    # shape_dict['output_shape'] is output_size
    predictor = CNNLSTMAttentionPredictor(
        input_channels=shape_dict['input_shape'][0], 
        seq_length=shape_dict['input_shape'][1],   
        lstm_hidden_size=64, # 与原CNN-LSTM一致或调整
        lstm_num_layers=1,   # 与原CNN-LSTM一致或调整
        output_size=shape_dict['output_shape'],
        cnn_out_channels=32, # CNN的输出通道数
        kernel_size=3        # CNN的卷积核大小
    )
    
    # 训练模型
    model_result, execution_time = predictor.train(train_loader, val_loader, epochs=epochs)
    
    # 预测
    predictions, actuals = predictor.predict(test_loader)
    
    # 反归一化预测结果
    predictions_original = inverse_transform(data_processor.scaler, predictions)
    actuals_original = inverse_transform(data_processor.scaler, actuals)
    
    # 评估模型
    evaluation_result = evaluate_model(actuals_original, predictions_original, "CNN-LSTM-Attention", execution_time)
    
    # 绘制结果
    plot_path = os.path.join("time_series/results", "cnn_lstm_attention_prediction.png")
    plot_results(actuals_original, predictions_original, "CNN-LSTM-Attention Model Prediction Results", plot_path)
    
    logger.info(f"CNN-LSTM-Attention模型完成，评估结果: RMSE={evaluation_result['rmse']:.4f}, MAE={evaluation_result['mae']:.4f}, R2={evaluation_result['r2']:.4f}")
    
    return evaluation_result

def main():
    """主函数"""
    global logger
    
    # 创建结果目录
    os.makedirs("time_series/results", exist_ok=True)
    os.makedirs("time_series/logs", exist_ok=True)
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("time_series/logs", f"time_series_prediction_{timestamp}.log")
    logger = setup_logger("time_series", log_file)
    
    # 配置其他模块的日志记录器
    for module in ['data_utils', 'data_processor', 'evaluation', 
                   'lstm_model', 'cnn_model', 'cnn_lstm_model', 'sklearn_models',
                   'arima_model', 'prophet_model', 
                   'transformer_model', 'cnn_lstm_attention_model']: # 添加新模型的日志
        module_logger = logging.getLogger(module)
        module_logger.handlers = [] 
        for handler in logger.handlers:
            module_logger.addHandler(handler)
        module_logger.setLevel(logger.level)
    
    logger.info("开始运行时间序列预测系统")
    logger.info(f"日志文件: {log_file}")
    
    # 加载数据
    file_path = os.path.join("time_series", "北京-逐日.xlsx")
    data_df_original = load_data(file_path) # 保存原始DataFrame副本
    
    if data_df_original is None or data_df_original.empty:
        logger.error("未能加载数据，程序终止。")
        return []

    # 准备数据 (用于深度学习和Sklearn模型)
    # prepare_data会进行归一化和序列化
    data_dict_processed = prepare_data(data_df_original.copy(), seq_length=7) # 使用副本以防修改原始df
    
    # 创建数据处理器
    data_processor = TimeSeriesDataProcessor(data_dict_processed, batch_size=32)
    
    # 收集所有模型的评估结果
    all_results = []
    
    # 运行原有模型
    current_epochs = 20 # 您可以根据需要调整这里的 epoch 值
    lstm_result = run_lstm_model(data_processor, epochs=current_epochs)
    all_results.append(lstm_result)
    cnn_result = run_cnn_model(data_processor, epochs=current_epochs)
    all_results.append(cnn_result)
    cnn_lstm_result = run_cnn_lstm_model(data_processor, epochs=current_epochs)
    all_results.append(cnn_lstm_result)
    sklearn_results = run_sklearn_models(data_processor)
    all_results.extend(sklearn_results)
    try:
        arima_result = run_arima_model(data_processor, data_df_original.copy())
        all_results.append(arima_result)
    except Exception as e:
        logger.error(f"运行ARIMA模型时发生错误: {e}")
        all_results.append({'model': "ARIMA", 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'mape': np.nan, 'execution_time': 0})
    try:
        prophet_result = run_prophet_model(data_processor, data_df_original.copy())
        all_results.append(prophet_result)
    except Exception as e:
        logger.error(f"运行Prophet模型时发生错误: {e}")
        all_results.append({'model': "Prophet", 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'mape': np.nan, 'execution_time': 0})

    # 运行新增的深度学习模型
    try:
        transformer_result = run_transformer_model(data_processor, epochs=current_epochs)
        all_results.append(transformer_result)
    except Exception as e:
        logger.error(f"运行Transformer模型时发生错误: {e}")
        all_results.append({'model': "Transformer", 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'mape': np.nan, 'execution_time': 0})
    
    try:
        cnn_lstm_attention_result = run_cnn_lstm_attention_model(data_processor, epochs=current_epochs)
        all_results.append(cnn_lstm_attention_result)
    except Exception as e:
        logger.error(f"运行CNN-LSTM-Attention模型时发生错误: {e}")
        all_results.append({'model': "CNN-LSTM-Attention", 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'mape': np.nan, 'execution_time': 0})

    # 保存所有结果
    excel_path = os.path.join("time_series/results", "all_models_comparison.xlsx")
    plot_path = os.path.join("time_series/results", "model_comparison_plot.png")
    
    # 过滤掉可能存在的None或评估失败的结果（以rmse为例）
    valid_results = [res for res in all_results if res and not (isinstance(res.get('rmse'), float) and np.isnan(res.get('rmse')))]
    if not valid_results:
        logger.warning("没有有效的模型评估结果可供保存。")
    else:
        results_df = save_all_results(valid_results, excel_path, plot_path)
        logger.info(f"所有有效结果已保存到 {excel_path}")

    logger.info("时间序列预测系统运行完成")
    
    return all_results # 返回所有尝试运行的结果，包括可能失败的

if __name__ == "__main__":
    main() 