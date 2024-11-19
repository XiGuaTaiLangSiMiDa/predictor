from prophet import Prophet
import pandas as pd
import numpy as np
from utils.data_cleaning import clean_dataframe

class ProphetModel:
    def __init__(self, daily_seasonality=True, yearly_seasonality=True):
        """
        初始化Prophet模型
        
        参数:
            daily_seasonality (bool): 是否包含日度季节性
            yearly_seasonality (bool): 是否包含年度季节性
        """
        self.model = None
        self.daily_seasonality = daily_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.regressors = ['rsi', 'volume', 'macd']
        self.last_regressor_values = None  # 存储最后的回归变量值
    
    def prepare_data(self, df):
        """
        准备Prophet模型的训练数据
        
        参数:
            df (pd.DataFrame): 输入数据
            
        返回:
            pd.DataFrame: 准备好的数据
        """
        print("\nPreparing Prophet data...")
        
        try:
            # 创建数据副本
            df_copy = df.copy()
            
            # 准备Prophet数据框
            prophet_df = pd.DataFrame()
            prophet_df['ds'] = df_copy.index
            prophet_df['y'] = df_copy['close'].values
            
            # 验证目标变量
            if prophet_df['y'].isna().any():
                print("Warning: NaN values found in target variable 'y', filling with forward fill")
                prophet_df['y'] = prophet_df['y'].ffill().bfill()
            
            print(f"Target variable range: [{prophet_df['y'].min():.2f}, {prophet_df['y'].max():.2f}]")
            
            # 添加回归变量
            for regressor in self.regressors:
                if regressor in df_copy.columns:
                    # 获取值并处理NaN/inf
                    values = df_copy[regressor].values
                    values = np.where(np.isinf(values), np.nan, values)
                    
                    # 计算中位数（排除NaN值）
                    median_value = np.nanmedian(values)
                    if np.isnan(median_value):
                        print(f"Warning: All values are NaN for {regressor}, using 0")
                        median_value = 0
                    
                    # 用中位数填充NaN值
                    values = np.where(np.isnan(values), median_value, values)
                    
                    # 添加到prophet数据框
                    prophet_df[regressor] = values
                    
                    print(f"Regressor {regressor}:")
                    print(f"  Range: [{np.min(values):.2f}, {np.max(values):.2f}]")
                    print(f"  Median: {median_value:.2f}")
                    print(f"  NaN values after filling: {np.isnan(values).sum()}")
                else:
                    print(f"Warning: Regressor {regressor} not found in data, using zeros")
                    prophet_df[regressor] = 0
            
            # 存储最后的回归变量值用于预测
            self.last_regressor_values = {
                regressor: prophet_df[regressor].iloc[-1]
                for regressor in self.regressors
                if regressor in prophet_df.columns
            }
            
            # 最终验证
            print("\nFinal Prophet dataframe shape:", prophet_df.shape)
            print("Columns:", prophet_df.columns.tolist())
            
            # 检查是否还有NaN值
            nan_counts = prophet_df.isna().sum()
            if nan_counts.any():
                print("\nWarning: Remaining NaN values:")
                print(nan_counts[nan_counts > 0])
                # 最后尝试清理任何剩余的NaN值
                prophet_df = prophet_df.ffill().bfill()
            
            print("Final NaN check:", prophet_df.isna().sum().sum() == 0)
            
            return prophet_df
            
        except Exception as e:
            print(f"Error in prepare_data: {e}")
            import traceback
            traceback.print_exc()
            raise

    def train(self, prophet_df):
        """
        训练Prophet模型
        
        参数:
            prophet_df (pd.DataFrame): 准备好的训练数据
        """
        try:
            # 验证数据
            nan_check = prophet_df.isna().sum()
            if nan_check.any():
                print("\nError: NaN values found in columns:")
                print(nan_check[nan_check > 0])
                raise ValueError("Prophet data contains NaN values")
            
            # 创建和配置模型
            self.model = Prophet(
                daily_seasonality=self.daily_seasonality,
                yearly_seasonality=self.yearly_seasonality,
                uncertainty_samples=1000
            )
            
            # 添加回归变量
            for regressor in self.regressors:
                if regressor in prophet_df.columns:
                    print(f"Adding regressor: {regressor}")
                    self.model.add_regressor(regressor)
            
            # 训练模型
            print("Training Prophet model...")
            self.model.fit(prophet_df)
            print("Prophet model training completed")
            return self.model
            
        except Exception as e:
            print(f"Error in train: {e}")
            import traceback
            traceback.print_exc()
            raise

    def predict(self, periods=30, freq='D', prophet_df=None):
        """
        生成预测
        
        参数:
            periods (int): 预测期数
            freq (str): 时间频率
            prophet_df (pd.DataFrame): 用于回归变量的历史数据
            
        返回:
            pd.DataFrame: 预测结果
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        try:
            # 创建未来数据框
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
            
            # 添加回归变量值
            if self.last_regressor_values:
                for regressor, last_value in self.last_regressor_values.items():
                    # 对所有未来时间点使用最后观察到的值
                    future[regressor] = last_value
            
            # 生成预测
            forecast = self.model.predict(future)
            return forecast
            
        except Exception as e:
            print(f"Error in predict: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_components(self):
        """获取趋势、季节性和节假日成分"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.component_modes

    def plot_components(self, forecast):
        """绘制预测成分"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        self.model.plot_components(forecast)

    def get_feature_importance(self):
        """获取每个回归变量的重要性"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        try:
            # 提取回归变量系数
            params = self.model.params
            regressor_coeffs = {}
            
            if 'beta' in params:
                for i, name in enumerate(self.regressors):
                    if i < len(params['beta']):
                        regressor_coeffs[name] = params['beta'][i]
            
            # 转换为DataFrame以便分析
            importance_df = pd.DataFrame({
                'regressor': list(regressor_coeffs.keys()),
                'coefficient': list(regressor_coeffs.values())
            })
            if not importance_df.empty:
                importance_df['abs_coefficient'] = abs(importance_df['coefficient'])
                importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
            
            return importance_df
            
        except Exception as e:
            print(f"Error in get_feature_importance: {e}")
            return pd.DataFrame()
