
## 安装部署

1. 克隆项目到本地：
```bash
git clone [repository-url]
cd time_series
```

2. 创建并激活虚拟环境（推荐）：
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

4. 准备数据：
- 将时间序列数据文件（Excel格式）放在项目根目录下
- 数据格式要求：
  - 必须包含"日期"列
  - 必须包含"AQI"列
  - Excel文件默认名称：`北京-逐日.xlsx`

5. 运行程序：
```bash
python main.py
```

## 输出说明

程序运行后会在以下目录生成结果：

- `results/`：
  - `all_models_comparison.xlsx`：所有模型的性能对比
  - `model_comparison_plot.png`：模型性能对比图
  - 各个模型的预测结果图

- `logs/`：
  - 详细的运行日志，包含时间戳

## 评估指标

系统使用以下指标评估模型性能：
- RMSE (均方根误差)
- MAE (平均绝对误差)
- R² (决定系数)
- MAPE (平均绝对百分比误差)
- 执行时间

## 自定义配置

在 `main.py` 中可以调整以下参数：

- `seq_length`：时间序列长度（默认为7）
- `batch_size`：批次大小（默认为32）
- `epochs`：训练轮数（默认为20）
- 各个模型的具体参数（如LSTM的隐藏层大小等）

## 注意事项

1. 确保安装了所有必要的依赖包
2. 数据文件必须符合指定格式
3. 对于大数据集，建议使用GPU进行训练
4. Prophet和ARIMA模型可能需要额外的系统依赖

## 问题排查

如果遇到问题：

1. 检查 `logs/` 目录下的日志文件
2. 确保数据格式正确
3. 检查是否所有依赖都已正确安装
4. 对于内存不足问题，可以调小batch_size

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。在提交PR之前，请确保：

1. 代码符合PEP 8规范
2. 添加了必要的注释
3. 更新了相关文档
4. 添加了必要的测试

## 许可证

[添加许可证信息]
