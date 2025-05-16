# 空气质量的时间序列预测系统

## 目录
- [项目简介](#项目简介)
- [项目结构](#项目结构)
- [安装部署](#安装部署)
  - [环境要求](#环境要求)
  - [安装步骤](#安装步骤)
  - [数据准备](#数据准备)
  - [运行程序](#运行程序)
- [模型详解](#模型详解)
  - [深度学习模型](#深度学习模型)
  - [机器学习模型](#机器学习模型)
  - [统计模型](#统计模型)
- [评价指标](#评价指标)
- [输出说明](#输出说明)
- [自定义配置](#自定义配置)
- [问题排查](#问题排查)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 项目简介

这是一个综合性的时间序列预测系统，集成了多种机器学习和深度学习模型，用于空气质量指数(AQI)的预测。该系统实现了多种先进的预测模型，并提供了完整的数据处理、模型训练、评估和可视化功能。

## 项目结构

```
time_series/
│
├── main.py                 # 主程序入口
├── requirements.txt        # 项目依赖
├── README.md              # 项目说明文档
│
├── models/                # 各种模型的实现
│   ├── lstm_model.py
│   ├── cnn_model.py
│   ├── cnn_lstm_model.py
│   ├── transformer_model.py
│   ├── cnn_lstm_attention_model.py
│   ├── sklearn_models.py
│   ├── arima_model.py
│   └── prophet_model.py
│
├── preprocessing/         # 数据预处理相关代码
│   └── data_processor.py
│
├── utils/                # 工具函数
│   ├── data_utils.py
│   └── evaluation.py
│
├── results/              # 模型预测结果和可视化
└── logs/                 # 运行日志
```

## 安装部署

### 环境要求
- Python 3.7+
- CUDA (可选，用于GPU加速)
- 足够的磁盘空间（建议>1GB）
- 足够的内存（建议>8GB）

### 安装步骤

1. 克隆项目到本地：
```bash
git clone [repository-url]
cd time_series
```

2. 创建并激活虚拟环境：
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

### 数据准备
- 将时间序列数据文件（Excel格式）放在项目根目录下
- 数据格式要求：
  - 必须包含"日期"列
  - 必须包含"AQI"列
  - Excel文件默认名称：`北京-逐日.xlsx`

### 运行程序
```bash
python main.py
```

## 模型详解

### 深度学习模型

#### 1. LSTM
- **特点**：能够捕获长期依赖关系
- **结构**：
  - 双层LSTM
  - 隐藏层大小64
  - Dropout防过拟合

#### 2. CNN
- **特点**：善于提取局部特征
- **结构**：
  - 多层卷积层
  - 池化层
  - 全连接层

#### 3. CNN-LSTM
- **特点**：结合CNN的特征提取和LSTM的序列建模能力
- **结构**：
  - CNN特征提取
  - LSTM序列建模
  - 全连接输出层

#### 4. Transformer
- **特点**：自注意力机制，并行计算
- **结构**：
  - 输入嵌入维度：64
  - 注意力头数：4
  - 编码器层数：2

#### 5. CNN-LSTM-Attention
- **特点**：注意力增强的混合模型
- **结构**：
  - CNN特征提取
  - LSTM序列处理
  - 注意力机制

### 机器学习模型

- **线性回归**：适用于线性趋势
- **决策树**：非线性关系建模
- **随机森林**：集成学习
- **XGBoost**：高效梯度提升
- **SVM**：支持向量机
- **梯度提升树**：序列化决策树
- **LightGBM**：轻量级梯度提升
- **CatBoost**：类别特征处理

### 统计模型

- **ARIMA**：经典时间序列模型
- **Prophet**：Facebook开发的分解模型

## 评价指标

### 1. 准确性指标

- **RMSE (均方根误差)**
  ```
  RMSE = sqrt(1/n * Σ(y_i - ŷ_i)²)
  ```

- **MAE (平均绝对误差)**
  ```
  MAE = 1/n * Σ|y_i - ŷ_i|
  ```

- **R² (决定系数)**
  ```
  R² = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²
  ```

- **MAPE (平均绝对百分比误差)**
  ```
  MAPE = 100% * 1/n * Σ|y_i - ŷ_i|/|y_i|
  ```

### 2. 效率指标
- 执行时间（秒）
- 内存使用
- GPU利用率（如适用）

## 输出说明

程序运行后会生成：

### results/目录
- `all_models_comparison.xlsx`：模型性能对比
- `model_comparison_plot.png`：可视化对比
- 各模型预测结果图

### logs/目录
- 详细运行日志
- 错误信息
- 性能指标

## 自定义配置

在 `main.py` 中可配置：
- `seq_length`：序列长度（默认7）
- `batch_size`：批次大小（默认32）
- `epochs`：训练轮数（默认20）
- 模型具体参数

## 许可证

MIT License

## 支持项目 ⭐

如果这个项目对您有帮助，请考虑给它一个星标 ⭐️ 这将帮助我们接触到更多的人，并激励我们继续改进项目。

## 联系与反馈 📮

- 如果您发现任何问题或有改进建议，请提交 Issue
- 欢迎通过 Pull Requests 贡献代码
- 如果您使用了这个项目，也欢迎分享您的使用案例

感谢您的支持！ 🙏 
