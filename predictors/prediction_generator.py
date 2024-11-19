import pandas as pd
import numpy as np
from utils.technical_indicators import TechnicalIndicators

class PredictionGenerator:
    """预测生成器类，处理所有预测生成相关的功能"""
    
    def __init__(self, lstm_model, prophet_model, rf_model, data_processor):
        """
        初始化预测生成器
        
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
    
    def generate_predictions(self, df, market_data, model_weights, lookback, periods=30):
        """
        生成预测结果
        
        参数:
            df (pd.DataFrame): 输入数据
            market_data (dict): 市场数据
            model_weights (dict): 模型权重
            lookback (int): 回看期长度
            periods (int): 预测期数
            
        返回:
            pd.DataFrame: 预测结果
        """
        try:
            print("\nGenerating predictions...")
            future_dates = pd.date_range(start=df.index[-1], periods=periods+1, freq='D')[1:]
            predictions = pd.DataFrame(index=future_dates)
            
            # 生成各个模型的预测
            model_predictions = self._generate_model_predictions(df, periods, lookback)
            
            # 将成功的预测添加到DataFrame
            for model, preds in model_predictions.items():
                if preds is not None:
                    predictions[model] = preds
            
            # 检查是否有有效预测
            if predictions.empty:
                raise ValueError("No valid predictions generated from any model")
            
            # 计算集成预测
            predictions = self._calculate_ensemble_predictions(predictions, model_weights)
            
            # 根据市场数据调整预测
            predictions = self._adjust_predictions_with_market_data(predictions, market_data)
            
            print("\nPrediction generation completed")
            return predictions
            
        except Exception as e:
            print(f"Error generating predictions: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_model_predictions(self, df, periods, lookback):
        """
        生成各个模型的预测
        
        参数:
            df (pd.DataFrame): 输入数据
            periods (int): 预测期数
            lookback (int): 回看期长度
            
        返回:
            dict: 各个模型的预测结果
        """
        model_predictions = {}
        
        try:
            # Prophet预测
            prophet_forecast = self.prophet_model.predict(periods=periods)
            if prophet_forecast is not None:
                model_predictions['prophet'] = prophet_forecast.tail(periods)['yhat'].values
                print("Prophet predictions generated")
        except Exception as e:
            print(f"Error generating Prophet predictions: {e}")
            model_predictions['prophet'] = None
        
        try:
            # LSTM预测
            feature_columns = self.data_processor.get_feature_columns()
            X_lstm, _, _ = self.data_processor.prepare_lstm_data(df, feature_columns, lookback)
            lstm_preds = self.lstm_model.predict_sequence(X_lstm[-1], periods)
            if lstm_preds is not None:
                model_predictions['lstm'] = lstm_preds.flatten()[:periods]
                print("LSTM predictions generated")
        except Exception as e:
            print(f"Error generating LSTM predictions: {e}")
            model_predictions['lstm'] = None
        
        try:
            # Random Forest预测
            feature_columns = self.data_processor.get_feature_columns()
            last_features = df[feature_columns].tail(lookback).values.flatten()
            rf_preds = self.rf_model.predict_sequence(last_features, periods)
            if rf_preds is not None:
                model_predictions['random_forest'] = rf_preds
                print("Random Forest predictions generated")
        except Exception as e:
            print(f"Error generating Random Forest predictions: {e}")
            model_predictions['random_forest'] = None
        
        return model_predictions
    
    def _calculate_ensemble_predictions(self, predictions, weights):
        """
        计算集成预测
        
        参数:
            predictions (pd.DataFrame): 各个模型的预测
            weights (dict): 模型权重
            
        返回:
            pd.DataFrame: 添加集成预测后的结果
        """
        # 初始化集成预测列
        predictions['ensemble'] = 0.0
        total_weight = 0.0
        
        # 添加每个成功模型的加权预测
        for model, weight in weights.items():
            if model in predictions.columns:
                predictions['ensemble'] += predictions[model] * weight
                total_weight += weight
        
        # 如果不是所有模型都成功，则归一化权重
        if total_weight > 0 and total_weight != 1.0:
            predictions['ensemble'] = predictions['ensemble'] / total_weight
        
        return predictions
    
    def _adjust_predictions_with_market_data(self, predictions, market_data):
        """
        根据市场数据调整预测
        
        参数:
            predictions (pd.DataFrame): 预测结果
            market_data (dict): 市场数据
            
        返回:
            pd.DataFrame: 调整后的预测结果
        """
        if market_data:
            # 恐慌贪婪指数调整
            if 'sentiment' in market_data and market_data['sentiment']:
                sentiment_data = market_data['sentiment']
                if 'fear_greed_index' in sentiment_data:
                    fear_greed_data = sentiment_data['fear_greed_index']
                    if isinstance(fear_greed_data, dict) and 'value' in fear_greed_data:
                        fear_greed_factor = float(fear_greed_data['value']) / 100
                        predictions['ensemble'] *= fear_greed_factor
            
            # 资金费率调整
            if 'derivatives' in market_data and market_data['derivatives']:
                derivatives_data = market_data['derivatives']
                if 'funding_rate' in derivatives_data and derivatives_data['funding_rate'] is not None:
                    funding_rate_factor = 1 + derivatives_data['funding_rate']
                    predictions['ensemble'] *= funding_rate_factor
        
        return predictions
    
    def get_prediction_intervals(self, predictions, confidence=0.95):
        """
        计算预测区间
        
        参数:
            predictions (pd.DataFrame): 预测结果
            confidence (float): 置信水平
            
        返回:
            tuple: (下界, 上界)
        """
        # 计算预测标准差
        std = predictions.drop('ensemble', axis=1).std(axis=1)
        
        # 计算置信区间
        z_score = 1.96  # 95% 置信水平
        lower_bound = predictions['ensemble'] - z_score * std
        upper_bound = predictions['ensemble'] + z_score * std
        
        return lower_bound, upper_bound
