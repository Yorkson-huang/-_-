import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import logging
from time_series.utils.evaluation import time_model_execution

class SklearnTimeSeriesModels:
    """
    基于Sklearn的时间序列预测模型集合
    """
    def __init__(self):
        self.logger = logging.getLogger('sklearn_models')
        self.models = {}
        self.logger.info("初始化Sklearn模型集合")
    
    @time_model_execution
    def train_random_forest(self, X_train, y_train, X_val=None, y_val=None, **params):
        """
        训练随机森林模型
        """
        # 默认参数
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        
        # 更新参数
        model_params = {**default_params, **params}
        self.logger.info(f"训练随机森林模型，参数: {model_params}")
        
        # 创建模型
        model = RandomForestRegressor(**model_params)
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 存储模型
        self.models['random_forest'] = model
        
        # 如果提供验证集，计算验证得分
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            self.logger.info(f"随机森林模型验证R²得分: {val_score:.4f}")
        
        return model
    
    @time_model_execution
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, **params):
        """
        训练XGBoost模型
        """
        # 默认参数
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        
        # 更新参数
        model_params = {**default_params, **params}
        self.logger.info(f"训练XGBoost模型，参数: {model_params}")
        
        # 创建模型
        model = xgb.XGBRegressor(**model_params)
        
        # 训练模型
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # 存储模型
        self.models['xgboost'] = model
        
        # 如果提供验证集，计算验证得分
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            self.logger.info(f"XGBoost模型验证R²得分: {val_score:.4f}")
        
        return model
    
    @time_model_execution
    def train_linear_regression(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练线性回归模型
        """
        self.logger.info("训练线性回归模型")
        
        # 创建模型
        model = LinearRegression()
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 存储模型
        self.models['linear_regression'] = model
        
        # 如果提供验证集，计算验证得分
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            self.logger.info(f"线性回归模型验证R²得分: {val_score:.4f}")
        
        return model
    
    @time_model_execution
    def train_decision_tree(self, X_train, y_train, X_val=None, y_val=None, **params):
        """
        训练决策树模型
        """
        # 默认参数
        default_params = {
            'max_depth': 10,
            'random_state': 42
        }
        
        # 更新参数
        model_params = {**default_params, **params}
        self.logger.info(f"训练决策树模型，参数: {model_params}")
        
        # 创建模型
        model = DecisionTreeRegressor(**model_params)
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 存储模型
        self.models['decision_tree'] = model
        
        # 如果提供验证集，计算验证得分
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            self.logger.info(f"决策树模型验证R²得分: {val_score:.4f}")
        
        return model
    
    @time_model_execution
    def train_svm(self, X_train, y_train, X_val=None, y_val=None, **params):
        """
        训练支持向量机 (SVM) 模型
        """
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1
        }
        model_params = {**default_params, **params}
        self.logger.info(f"训练SVM模型，参数: {model_params}")
        model = SVR(**model_params)
        model.fit(X_train, y_train)
        self.models['svm'] = model
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            self.logger.info(f"SVM模型验证R²得分: {val_score:.4f}")
        return model

    @time_model_execution
    def train_gradient_boosting(self, X_train, y_train, X_val=None, y_val=None, **params):
        """
        训练梯度提升树 (Gradient Boosting) 模型
        """
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
        model_params = {**default_params, **params}
        self.logger.info(f"训练Gradient Boosting模型，参数: {model_params}")
        model = GradientBoostingRegressor(**model_params)
        model.fit(X_train, y_train)
        self.models['gradient_boosting'] = model
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            self.logger.info(f"Gradient Boosting模型验证R²得分: {val_score:.4f}")
        return model

    @time_model_execution
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None, **params):
        """
        训练LightGBM模型
        """
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': -1, # -1 表示没有限制
            'random_state': 42,
            'verbose': -1 # 控制LightGBM的日志输出等级
        }
        model_params = {**default_params, **params}
        self.logger.info(f"训练LightGBM模型，参数: {model_params}")
        model = lgb.LGBMRegressor(**model_params)
        
        eval_set = [(X_train, y_train)]
        callbacks = []
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            # LightGBM 使用 callbacks 来实现早停等功能，这里暂时不加复杂的早停
            # callbacks.append(lgb.early_stopping(10, verbose=False))

        model.fit(X_train, y_train,
                  eval_set=eval_set,
                  # callbacks=callbacks # 如果需要早停，取消注释此行
                  )
        self.models['lightgbm'] = model
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val) # score 方法通常是在 fit 之后直接可用的
            self.logger.info(f"LightGBM模型验证R²得分: {val_score:.4f}")
        return model

    @time_model_execution
    def train_catboost(self, X_train, y_train, X_val=None, y_val=None, **params):
        """
        训练CatBoost模型
        """
        default_params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'random_state': 42,
            'verbose': 0 # 控制CatBoost的日志输出等级
        }
        model_params = {**default_params, **params}
        self.logger.info(f"训练CatBoost模型，参数: {model_params}")
        model = cb.CatBoostRegressor(**model_params)
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        model.fit(X_train, y_train,
                  eval_set=eval_set,
                  # early_stopping_rounds=10 # 如果需要早停
                  )
        self.models['catboost'] = model
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            self.logger.info(f"CatBoost模型验证R²得分: {val_score:.4f}")
        return model

    def predict(self, model_name, X):
        """
        使用指定模型进行预测
        """
        if model_name not in self.models:
            self.logger.error(f"模型 '{model_name}' 不存在")
            raise ValueError(f"模型 '{model_name}' 不存在")
        
        self.logger.info(f"使用 {model_name} 模型进行预测")
        
        # 进行预测
        predictions = self.models[model_name].predict(X)
        
        return predictions
    
    def get_model(self, model_name):
        """
        获取指定名称的模型
        """
        if model_name not in self.models:
            self.logger.error(f"模型 '{model_name}' 不存在")
            raise ValueError(f"模型 '{model_name}' 不存在")
        
        return self.models[model_name] 