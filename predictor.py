from predictors.base_predictor import BasePredictor
from predictors.data_processor import DataProcessor
from predictors.model_trainer import ModelTrainer
from predictors.prediction_generator import PredictionGenerator
from predictors.market_analyzer import MarketAnalyzer

class CryptoPredictor(BasePredictor):
    """加密货币价格预测器，整合所有预测和分析功能"""
    
    def __init__(self, symbol='BTC/USDT'):
        """
        初始化预测器
        
        参数:
            symbol (str): 交易对符号，默认为'BTC/USDT'
        """
        super().__init__(symbol)
        
        # 初始化各个组件
        self.data_processor = DataProcessor(self.data_fetcher)
        self.model_trainer = ModelTrainer(
            self.lstm_model,
            self.prophet_model,
            self.rf_model,
            self.data_processor
        )
        self.prediction_generator = PredictionGenerator(
            self.lstm_model,
            self.prophet_model,
            self.rf_model,
            self.data_processor
        )
        self.market_analyzer = MarketAnalyzer(
            self.data_fetcher,
            self.data_processor
        )
    
    def prepare_data(self):
        """获取和准备预测数据"""
        return self.data_processor.prepare_data()
    
    def train_models(self, df):
        """训练所有模型"""
        return self.model_trainer.train_models(df, self.lookback)
    
    def generate_predictions(self, periods=30):
        """生成预测结果"""
        try:
            # 准备数据
            df, signals = self.prepare_data()
            if df is None:
                raise ValueError("Failed to prepare data")
            
            # 训练模型
            lstm_data, prophet_data = self.train_models(df)
            if lstm_data is None:
                raise ValueError("Failed to train models")
            
            # 获取市场数据
            market_data = self.data_fetcher.get_market_summary()
            
            # 生成预测
            predictions = self.prediction_generator.generate_predictions(
                df,
                market_data,
                self.model_weights,
                self.lookback,
                periods
            )
            if predictions is None:
                raise ValueError("Failed to generate predictions")
            
            return predictions, df, signals
            
        except Exception as e:
            print(f"Error generating predictions: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def analyze_market(self):
        """执行市场分析"""
        try:
            # 生成预测
            predictions, df, signals = self.generate_predictions()
            if predictions is None:
                return None
            
            # 分析市场
            analysis_results = self.market_analyzer.analyze_market(
                predictions,
                df,
                signals
            )
            
            return analysis_results
            
        except Exception as e:
            print(f"Error in market analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_model_insights(self):
        """获取模型洞察"""
        return self.model_trainer.get_model_insights()
