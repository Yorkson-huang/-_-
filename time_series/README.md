# 时间序列预测系统

本项目实现了一个综合性的时间序列预测系统，用于对北京市AQI数据进行预测分析。系统包含多种预测模型，并提供了详细的评估指标和可视化结果。

## 项目结构

```
time_series/
├── models/             # 模型实现
│   ├── lstm_model.py   # LSTM模型
│   ├── cnn_model.py    # CNN模型
│   ├── cnn_lstm_model.py # CNN+LSTM组合模型
│   └── sklearn_models.py # 传统机器学习模型(随机森林、XGBoost等)
├── preprocessing/      # 数据预处理
│   └── data_processor.py # 数据处理器
├── utils/              # 工具函数
│   ├── data_utils.py   # 数据工具
│   └── evaluation.py   # 评估工具
├── evaluation/         # 模型评估
├── logs/               # 运行日志
├── results/            # 结果输出
├── main.py             # 主程序
├── requirements.txt    # 依赖包
└── 北京-逐日.xlsx      # 数据集
```

## 支持的模型

系统支持以下预测模型：

1. **深度学习模型**:
   - LSTM (长短期记忆网络)
   - CNN (卷积神经网络)
   - CNN+LSTM (组合模型)

2. **传统机器学习模型**:
   - 线性回归
   - 决策树
   - 随机森林
   - XGBoost

## 评估指标

系统使用以下指标评估模型性能：

- RMSE (均方根误差)
- MAE (平均绝对误差)
- R² (决定系数)
- MAPE (平均绝对百分比误差)
- 执行时间

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方法

```bash
cd time_series
python main.py
```

## 输出结果

系统会在`time_series/results`目录下生成以下内容：

1. 各个模型的预测曲线图
2. 所有模型性能对比图
3. 以Excel格式保存的完整性能评估结果

系统还会在`time_series/logs`目录下生成详细的运行日志。

## 中文支持

系统完全支持中文，包括日志、图表标签、评估结果等。 