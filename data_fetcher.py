from data.base_fetcher import BaseFetcher
from data.onchain_fetcher import OnchainFetcher
from data.derivatives_fetcher import DerivativesFetcher
from data.sentiment_fetcher import SentimentFetcher

class DataFetcher:
    """主数据获取类，整合所有数据源"""
    
    def __init__(self, spot_symbol='BTC/USDT', futures_symbol='BTC/USDT:USDT',
                 glassnode_api_key='', twitter_api_key=''):
        """
        初始化数据获取器
        
        参数:
            spot_symbol (str): 现货交易对
            futures_symbol (str): 合约交易对
            glassnode_api_key (str): Glassnode API密钥
            twitter_api_key (str): Twitter API密钥
        """
        self.base_currency = spot_symbol.split('/')[0]  # 'BTC'
        
        # 初始化各个专门的数据获取器
        self.base_fetcher = BaseFetcher(spot_symbol, futures_symbol)
        self.onchain_fetcher = OnchainFetcher(self.base_currency, glassnode_api_key)
        self.derivatives_fetcher = DerivativesFetcher(spot_symbol, futures_symbol)
        self.sentiment_fetcher = SentimentFetcher(self.base_currency, twitter_api_key)
    
    def fetch_historical_data(self, timeframe='1h', limit=1000):
        """获取历史价格数据"""
        return self.base_fetcher.fetch_historical_data(timeframe, limit)
    
    def fetch_market_data(self):
        """
        获取完整的市场数据
        
        返回:
            dict: 包含所有市场数据的字典
        """
        market_data = {}
        
        # 获取基本市场数据
        market_summary = self.base_fetcher.get_market_summary()
        if market_summary:
            market_data['market_summary'] = market_summary
        
        # 获取链上数据
        onchain_data = self.onchain_fetcher.get_all_onchain_data()
        if onchain_data:
            market_data['onchain'] = onchain_data
        
        # 获取衍生品市场数据
        derivatives_data = self.derivatives_fetcher.get_all_derivatives_data()
        if derivatives_data:
            market_data['derivatives'] = derivatives_data
        
        # 获取市场情绪数据
        sentiment_data = self.sentiment_fetcher.get_all_sentiment_data()
        if sentiment_data:
            market_data['sentiment'] = sentiment_data
        
        return market_data
    
    def get_market_summary(self):
        """
        获取市场概况
        
        返回:
            dict: 市场概况数据
        """
        return self.fetch_market_data()
    
    def validate_symbol(self):
        """验证交易对是否有效"""
        return self.base_fetcher.validate_symbol()
    
    def get_data_status(self):
        """
        获取数据源状态
        
        返回:
            dict: 包含各个数据源状态的字典
        """
        status = {
            'base_data': True,  # 基础数据总是可用
            'onchain_data': bool(self.onchain_fetcher.glassnode_api_key),
            'derivatives_data': True,  # 衍生品数据总是可用
            'sentiment_data': {
                'fear_greed': True,  # 恐慌贪婪指数总是可用
                'social_media': bool(self.sentiment_fetcher.twitter_api_key),
                'google_trends': True,  # Google趋势总是可用
            }
        }
        return status
    
    def get_available_timeframes(self):
        """
        获取可用的时间周期
        
        返回:
            list: 可用的时间周期列表
        """
        return [
            '1m', '3m', '5m', '15m', '30m',  # 分钟
            '1h', '2h', '4h', '6h', '8h', '12h',  # 小时
            '1d', '3d',  # 天
            '1w',  # 周
            '1M'   # 月
        ]
    
    def get_available_indicators(self):
        """
        获取可用的指标
        
        返回:
            dict: 包含可用指标的字典
        """
        return {
            'technical': [
                'MA', 'EMA', 'RSI', 'MACD',
                'Bollinger Bands', 'Volume',
                'OBV', 'ATR', 'Volatility'
            ],
            'onchain': [
                'Whale Holdings', 'Exchange Flows',
                'Active Addresses', 'Miner Data'
            ],
            'derivatives': [
                'Funding Rate', 'Premium Rate',
                'Open Interest', 'Long/Short Ratio'
            ],
            'sentiment': [
                'Fear & Greed Index', 'Social Media',
                'Google Trends', 'News Sentiment'
            ]
        }
