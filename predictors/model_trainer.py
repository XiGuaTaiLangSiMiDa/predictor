import numpy as np
import pandas as pd

class ModelTrainer:
    """模型训练器类，处理所有模型训练相关的功能"""
    
    def __init__(self, lstm_model, prophet_model, rf_model, data_processor):
        """
        初始化模型训练器
        
        参数:
            lstm_model (LSTMModel): LSTM模型实例
            prophet_model (ProphetModel): Prophet模型实例
            rf_model (RandomForestModel): Random Forest模型实例
            data_processor (DataProcessor): 数据处理器实例
        """
        self.lstm_model = lstm_model
        self.prophet_model = prophet_model
        self.rf_model = rf_model
        self.data_processor = data_processor
    
    def train_models(self, df, lookback):
        """
        训练所有模型
        
        参数:
            df (pd.DataFrame): 输入数据
            lookback (int): 回看期长度
            
        返回:
            tuple: (lstm_data, prophet_data)
        """
        print("\nTraining models...")
        
        try:
            # 准备特征列表
            feature_columns = self.data_processor.get_feature_columns()
            
            # 训练LSTM
            lstm_data = self._train_lstm(df, feature_columns, lookback)
            
            # 训练Prophet
            prophet_data = self._train_prophet(df)
            
            # 训练Random Forest
            self._train_random_forest(df, feature_columns, lookback)
            
            return lstm_data, prophet_data
            
        except Exception as e:
            print(f"Error in train_models: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _train_lstm(self, df, feature_columns, lookback):
        """
        训练LSTM模型
        
        参数:
            df (pd.DataFrame): 输入数据
            feature_columns (list): 特征列名列表
            lookback (int): 回看期长度
            
        返回:
            tuple: (X_lstm, y_lstm, n_features)
        """
        try:
            print("\nTraining LSTM model...")
            
            # 准备数据
            X_lstm, y_lstm, n_features = self.data_processor.prepare_lstm_data(
                df, feature_columns, lookback
            )
            
            # 构建模型
            self.lstm_model.build_model(n_features)
            
            # 训练模型
            self.lstm_model.train(X_lstm, y_lstm)
            print("LSTM training completed")
            
            return X_lstm
            
        except Exception as e:
            print(f"Error training LSTM model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _train_prophet(self, df):
        """
        训练Prophet模型
        
        参数:
            df (pd.DataFrame): 输入数据
            
        返回:
            pd.DataFrame: Prophet格式的数据
        """
        try:
            print("\nTraining Prophet model...")
            
            # 准备数据
            prophet_df = self.data_processor.prepare_prophet_data(df)
            
            # 训练模型
            self.prophet_model.train(prophet_df)
            print("Prophet training completed")
            
            return prophet_df
            
        except Exception as e:
            print(f"Error training Prophet model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _train_random_forest(self, df, feature_columns, lookback):
        """
        训练Random Forest模型
        
        参数:
            df (pd.DataFrame): 输入数据
            feature_columns (list): 特征列名列表
            lookback (int): 回看期长度
        """
        try:
            print("\nTraining Random Forest model...")
            
            # 准备数据
            X_rf, y_rf = self.data_processor.prepare_rf_data(
                df, feature_columns, lookback
            )
            
            # 设置特征列
            self.rf_model.feature_columns = feature_columns
            
            # 训练模型
            self.rf_model.train(X_rf, y_rf)
            print("Random Forest training completed")
            
        except Exception as e:
            print(f"Error training Random Forest model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_model_insights(self):
        """
        获取模型洞察
        
        返回:
            dict: 包含各个模型洞察的字典
        """
        insights = {}
        
        try:
            # LSTM洞察
            if hasattr(self, 'X_lstm'):
                insights['lstm'] = self.lstm_model.get_feature_importance(self.X_lstm)
        except Exception as e:
            print(f"Error getting LSTM insights: {e}")
        
        try:
            # Prophet洞察
            if self.prophet_model.model:
                insights['prophet'] = self.prophet_model.get_feature_importance()
        except Exception as e:
            print(f"Error getting Prophet insights: {e}")
        
        try:
            # Random Forest洞察
            if self.rf_model.model:
                insights['random_forest'] = self.rf_model.get_feature_importance()
        except Exception as e:
            print(f"Error getting Random Forest insights: {e}")
        
        return insights
    
    def get_training_status(self):
        """
        获取模型训练状态
        
        返回:
            dict: 包含各个模型训练状态的字典
        """
        status = {
            'lstm': hasattr(self.lstm_model, 'model') and self.lstm_model.model is not None,
            'prophet': hasattr(self.prophet_model, 'model') and self.prophet_model.model is not None,
            'random_forest': hasattr(self.rf_model, 'model') and self.rf_model.model is not None
        }
        return status
