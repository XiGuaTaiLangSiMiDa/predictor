import pandas as pd
import numpy as np
from utils.technical_indicators import TechnicalIndicators
from utils.data_cleaning import clean_dataframe, validate_data, check_data_quality, print_data_quality_report

class DataProcessor:
    """数据处理器类，处理所有数据准备和清理相关的功能"""
    
    def __init__(self, data_fetcher):
        """
        初始化数据处理器
        
        参数:
            data_fetcher (DataFetcher): 数据获取器实例
        """
        self.data_fetcher = data_fetcher
    
    def prepare_data(self):
        """
        获取和准备预测数据
        
        返回:
            tuple: (处理后的数据, 交易信号)
        """
        print("\nPreparing data for prediction...")
        
        try:
            # 获取历史价格数据
            df = self.data_fetcher.fetch_historical_data()
            if df is None:
                raise ValueError("Failed to fetch historical data")
            
            # 检查初始数据质量
            initial_quality = check_data_quality(df)
            print("\nInitial data quality report:")
            print_data_quality_report(initial_quality)
            
            # 添加技术指标
            df = self._add_technical_indicators(df)
            
            # 生成交易信号
            signals = self._generate_trading_signals(df)
            df = pd.concat([df, signals], axis=1)
            
            # 清理数据
            df = clean_dataframe(df)
            
            # 检查最终数据质量
            final_quality = check_data_quality(df)
            print("\nFinal data quality report:")
            print_data_quality_report(final_quality)
            
            return df, signals
            
        except Exception as e:
            print(f"Error in prepare_data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _add_technical_indicators(self, df):
        """
        添加技术指标
        
        参数:
            df (pd.DataFrame): 输入数据
            
        返回:
            pd.DataFrame: 添加技术指标后的数据
        """
        return TechnicalIndicators.add_all_indicators(df)
    
    def _generate_trading_signals(self, df):
        """
        生成交易信号
        
        参数:
            df (pd.DataFrame): 输入数据
            
        返回:
            pd.DataFrame: 交易信号
        """
        return TechnicalIndicators.generate_trading_signals(df)
    
    def prepare_lstm_data(self, df, feature_columns, lookback):
        """
        准备LSTM模型的训练数据
        
        参数:
            df (pd.DataFrame): 输入数据
            feature_columns (list): 特征列名列表
            lookback (int): 回看期长度
            
        返回:
            tuple: (X, y, n_features)
        """
        try:
            # 验证特征列
            missing_columns = [col for col in feature_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")
            
            # 准备数据
            data = df[feature_columns].values
            X, y = [], []
            
            for i in range(lookback, len(data)):
                X.append(data[i-lookback:i])
                y.append(data[i, 0])  # 第一列是收盘价
            
            X = np.array(X)
            y = np.array(y)
            
            print(f"Prepared LSTM data shapes - X: {X.shape}, y: {y.shape}")
            return X, y, len(feature_columns)
            
        except Exception as e:
            print(f"Error preparing LSTM data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def prepare_prophet_data(self, df):
        """
        准备Prophet模型的训练数据
        
        参数:
            df (pd.DataFrame): 输入数据
            
        返回:
            pd.DataFrame: Prophet格式的数据
        """
        try:
            # 创建Prophet数据框
            prophet_df = pd.DataFrame()
            prophet_df['ds'] = df.index
            prophet_df['y'] = df['close'].values
            
            # 添加回归变量
            for col in ['rsi', 'volume', 'macd']:
                if col in df.columns:
                    prophet_df[col] = df[col].values
            
            return prophet_df
            
        except Exception as e:
            print(f"Error preparing Prophet data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def prepare_rf_data(self, df, feature_columns, lookback):
        """
        准备Random Forest模型的训练数据
        
        参数:
            df (pd.DataFrame): 输入数据
            feature_columns (list): 特征列名列表
            lookback (int): 回看期长度
            
        返回:
            tuple: (X, y)
        """
        try:
            # 验证特征列
            missing_columns = [col for col in feature_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")
            
            # 准备数据
            data = df[feature_columns].values
            X, y = [], []
            
            for i in range(lookback, len(data)):
                X.append(data[i-lookback:i].flatten())  # 展平特征数据
                y.append(data[i, 0])  # 第一列是收盘价
            
            X = np.array(X)
            y = np.array(y)
            
            print(f"Prepared Random Forest data shapes - X: {X.shape}, y: {y.shape}")
            
            # 返回特征列名和数据
            return X, y
            
        except Exception as e:
            print(f"Error preparing Random Forest data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_feature_columns(self):
        """
        获取特征列名列表
        
        返回:
            list: 特征列名列表
        """
        return TechnicalIndicators.get_feature_columns()
