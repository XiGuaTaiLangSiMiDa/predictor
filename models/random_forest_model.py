import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from utils.data_cleaning import clean_dataframe

class RandomForestModel:
    def __init__(self, n_estimators=200, random_state=42, lookback=24):
        """
        初始化Random Forest模型
        
        参数:
            n_estimators (int): 决策树数量
            random_state (int): 随机种子
            lookback (int): 用于预测的历史数据点数量
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.lookback = lookback
        self.feature_columns = None
        self.feature_scaler = None
    
    def prepare_data(self, df, features, target='close'):
        """
        准备Random Forest模型的训练数据
        
        参数:
            df (pd.DataFrame): 输入数据
            features (list): 特征列名列表
            target (str): 目标变量列名
            
        返回:
            tuple: (X, y)
        """
        try:
            # 清理数据
            df = clean_dataframe(df)
            self.feature_columns = features
            
            print("\nPreparing Random Forest data...")
            print(f"Features: {features}")
            print(f"Target: {target}")
            
            X, y = [], []
            for i in range(self.lookback, len(df)):
                X.append(df[features].iloc[i-self.lookback:i].values.flatten())
                y.append(df[target].iloc[i])
            
            X = np.array(X)
            y = np.array(y)
            
            print(f"Prepared Random Forest data shapes - X: {X.shape}, y: {y.shape}")
            return X, y
            
        except Exception as e:
            print(f"Error preparing Random Forest data: {e}")
            import traceback
            traceback.print_exc()
            raise

    def train(self, X, y):
        """
        训练Random Forest模型
        
        参数:
            X (np.array): 训练特征
            y (np.array): 训练目标
        """
        try:
            print("\nTraining Random Forest model...")
            print(f"Training samples: {len(X)}")
            
            self.model.fit(X, y)
            
            # 计算并打印特征重要性
            importance = self.get_feature_importance()
            if importance is not None and not importance.empty:
                print("\nTop 10 most important features:")
                print(importance.head(10))
            
            print("Random Forest training completed")
            return self.model
            
        except Exception as e:
            print(f"Error training Random Forest model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def predict(self, X):
        """
        生成预测
        
        参数:
            X (np.array): 输入特征
            
        返回:
            np.array: 预测结果
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)

    def predict_sequence(self, last_features, n_steps):
        """
        生成预测序列
        
        参数:
            last_features (np.array): 最后的输入特征
            n_steps (int): 预测步数
            
        返回:
            np.array: 预测序列
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.feature_columns is None:
            raise ValueError("Feature columns not set. Call prepare_data first.")
        
        try:
            predictions = []
            current_features = last_features.copy()
            n_features = len(self.feature_columns)
            
            for _ in range(n_steps):
                # 重塑特征以进行预测
                input_features = current_features.reshape(1, -1)
                next_pred = self.model.predict(input_features)
                predictions.append(next_pred[0])
                
                # 更新特征用于下一次预测
                current_features = np.roll(current_features, -n_features)
                # 用预测值更新最新的特征集
                current_features[-n_features:] = np.array([next_pred[0]] + [0] * (n_features - 1))
            
            return np.array(predictions)
            
        except Exception as e:
            print(f"Error generating Random Forest predictions: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_feature_importance(self):
        """
        获取特征重要性
        
        返回:
            pd.DataFrame: 特征重要性排序
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.feature_columns is None:
            raise ValueError("Feature columns not set. Call prepare_data first.")
        
        try:
            # 获取特征重要性分数
            importance = self.model.feature_importances_
            
            # 为所有回看期创建特征名称
            feature_names = []
            for i in range(self.lookback):
                for feature in self.feature_columns:
                    feature_names.append(f"{feature}_t-{self.lookback-i}")
            
            # 创建包含特征名称和重要性分数的DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            return pd.DataFrame()

    def get_top_features(self, n=10):
        """
        获取最重要的n个特征
        
        参数:
            n (int): 返回的特征数量
            
        返回:
            pd.DataFrame: 前n个最重要的特征
        """
        importance_df = self.get_feature_importance()
        if importance_df is not None and not importance_df.empty:
            return importance_df.head(n)
        return pd.DataFrame()

    def plot_feature_importance(self, n=20):
        """
        绘制特征重要性
        
        参数:
            n (int): 显示的特征数量
        """
        import matplotlib.pyplot as plt
        
        importance_df = self.get_feature_importance()
        if importance_df is not None and not importance_df.empty:
            top_n = importance_df.head(n)
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(top_n)), top_n['importance'])
            plt.xticks(range(len(top_n)), top_n['feature'], rotation=45, ha='right')
            plt.title(f'Top {n} Feature Importance')
            plt.tight_layout()
            plt.savefig('random_forest_feature_importance.png')
            plt.close()

    def get_model_params(self):
        """
        获取模型参数和统计信息
        
        返回:
            dict: 模型参数
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        params = {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'n_features': len(self.feature_columns) * self.lookback if self.feature_columns else None,
            'lookback_period': self.lookback
        }
        
        return params
