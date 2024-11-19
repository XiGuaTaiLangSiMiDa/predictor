from data_fetcher import DataFetcher
from models.lstm_model import LSTMModel
from models.prophet_model import ProphetModel
from models.random_forest_model import RandomForestModel

class BasePredictor:
    """基础预测器类，包含所有预测器共享的基本属性和方法"""
    
    def __init__(self, symbol='BTC/USDT'):
        """
        初始化基础预测器
        
        参数:
            symbol (str): 交易对符号，默认为'BTC/USDT'
        """
        self.spot_symbol = symbol
        self.futures_symbol = f"{symbol}:USDT"
        self.data_fetcher = DataFetcher(self.spot_symbol, self.futures_symbol)
        
        # 初始化模型
        self.lstm_model = LSTMModel()
        self.prophet_model = ProphetModel()
        self.rf_model = RandomForestModel()
        
        # 预测参数
        self.lookback = 24
        self.model_weights = {
            'prophet': 0.3,
            'lstm': 0.4,
            'random_forest': 0.3
        }
    
    def validate_data(self, df):
        """
        验证数据有效性
        
        参数:
            df (pd.DataFrame): 输入数据
            
        返回:
            bool: 数据是否有效
        """
        if df is None or df.empty:
            print("Error: Empty or None DataFrame")
            return False
        
        required_columns = ['close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
        
        if df.isna().any().any():
            print("Warning: DataFrame contains NaN values")
            return False
        
        return True
    
    def validate_predictions(self, predictions):
        """
        验证预测结果有效性
        
        参数:
            predictions (pd.DataFrame): 预测结果
            
        返回:
            bool: 预测是否有效
        """
        if predictions is None or predictions.empty:
            print("Error: Empty or None predictions")
            return False
        
        if 'ensemble' not in predictions.columns:
            print("Error: Missing ensemble predictions")
            return False
        
        if predictions.isna().any().any():
            print("Warning: Predictions contain NaN values")
            return False
        
        return True
    
    def get_model_weights(self):
        """
        获取模型权重
        
        返回:
            dict: 模型权重字典
        """
        return self.model_weights.copy()
    
    def set_model_weights(self, weights):
        """
        设置模型权重
        
        参数:
            weights (dict): 新的权重字典
        """
        if not isinstance(weights, dict):
            raise ValueError("Weights must be a dictionary")
        
        if not all(model in self.model_weights for model in weights):
            raise ValueError("Invalid model names in weights")
        
        if not abs(sum(weights.values()) - 1.0) < 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        self.model_weights = weights.copy()
    
    def get_available_models(self):
        """
        获取可用模型列表
        
        返回:
            list: 可用模型名称列表
        """
        return list(self.model_weights.keys())
    
    def get_lookback_period(self):
        """
        获取回看期长度
        
        返回:
            int: 回看期长度
        """
        return self.lookback
    
    def set_lookback_period(self, lookback):
        """
        设置回看期长度
        
        参数:
            lookback (int): 新的回看期长度
        """
        if not isinstance(lookback, int) or lookback <= 0:
            raise ValueError("Lookback period must be a positive integer")
        
        self.lookback = lookback
        self.lstm_model = LSTMModel(lookback=lookback)
        self.rf_model = RandomForestModel(lookback=lookback)
    
    def get_model_parameters(self):
        """
        获取所有模型的参数
        
        返回:
            dict: 模型参数字典
        """
        params = {
            'lstm': {
                'lookback': self.lstm_model.lookback,
                'n_features': self.lstm_model.n_features if hasattr(self.lstm_model, 'n_features') else None
            },
            'prophet': {
                'daily_seasonality': self.prophet_model.daily_seasonality,
                'yearly_seasonality': self.prophet_model.yearly_seasonality,
                'regressors': self.prophet_model.regressors
            },
            'random_forest': self.rf_model.get_model_params() if hasattr(self.rf_model, 'model') else None
        }
        return params
    
    def get_data_fetcher(self):
        """
        获取数据获取器实例
        
        返回:
            DataFetcher: 数据获取器实例
        """
        return self.data_fetcher
