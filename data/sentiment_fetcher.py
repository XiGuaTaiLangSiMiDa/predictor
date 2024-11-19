import requests
from datetime import datetime
from pytrends.request import TrendReq

class SentimentFetcher:
    """市场情绪数据获取类，处理所有情绪相关数据"""
    
    def __init__(self, base_currency='BTC', twitter_api_key=''):
        """
        初始化情绪数据获取器
        
        参数:
            base_currency (str): 基础货币，如 'BTC'
            twitter_api_key (str): Twitter API密钥
        """
        self.base_currency = base_currency
        self.twitter_api_key = twitter_api_key
        self.pytrends = TrendReq(hl='en-US', tz=360)
    
    def fetch_fear_greed_index(self):
        """
        获取恐慌贪婪指数
        
        返回:
            dict: 包含恐慌贪婪指数数据的字典
        """
        try:
            response = requests.get('https://api.alternative.me/fng/')
            if response.status_code == 200:
                data = response.json()
                value = int(data['data'][0]['value'])
                classification = data['data'][0]['value_classification']
                return {
                    'value': value,
                    'classification': classification,
                    'timestamp': datetime.now().isoformat()
                }
            return None
        except Exception as e:
            print(f"Error fetching fear and greed index: {e}")
            return None
    
    def fetch_social_media_sentiment(self):
        """
        获取社交媒体情绪数据
        
        返回:
            dict: 包含社交媒体情绪数据的字典
        """
        if not self.twitter_api_key:
            print("Warning: Twitter API key not set")
            return None
            
        try:
            # 这里需要实现Twitter API的调用
            # 由于需要Twitter API密钥，这里只返回示例数据
            return {
                'twitter_sentiment': None,
                'tweet_volume': None,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error fetching social media sentiment: {e}")
            return None
    
    def fetch_google_trends(self):
        """
        获取Google趋势数据
        
        返回:
            dict: 包含Google趋势数据的字典
        """
        try:
            # 设置关键词
            kw_list = [
                self.base_currency,
                f"{self.base_currency} price",
                "cryptocurrency",
                "crypto"
            ]
            
            # 获取实时趋势数据
            self.pytrends.build_payload(kw_list, timeframe='now 1-d')
            interest_over_time = self.pytrends.interest_over_time()
            
            if not interest_over_time.empty:
                trends_data = {
                    'currency_trend': interest_over_time[self.base_currency].iloc[-1],
                    'price_trend': interest_over_time[f"{self.base_currency} price"].iloc[-1],
                    'crypto_trend': interest_over_time["cryptocurrency"].iloc[-1],
                    'timestamp': datetime.now().isoformat()
                }
                return trends_data
            return None
        except Exception as e:
            print(f"Error fetching Google Trends data: {e}")
            return None
    
    def fetch_reddit_sentiment(self):
        """
        获取Reddit情绪数据
        
        返回:
            dict: 包含Reddit情绪数据的字典
        """
        try:
            # 这里需要实现Reddit API的调用
            # 由于需要Reddit API密钥，这里只返回示例数据
            return {
                'reddit_sentiment': None,
                'post_volume': None,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error fetching Reddit sentiment: {e}")
            return None
    
    def analyze_news_sentiment(self):
        """
        分析新闻情绪
        
        返回:
            dict: 包含新闻情绪分析的字典
        """
        try:
            # 这里需要实现新闻API的调用和情绪分析
            # 由于需要新闻API密钥，这里只返回示例数据
            return {
                'news_sentiment': None,
                'news_volume': None,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error analyzing news sentiment: {e}")
            return None
    
    def get_all_sentiment_data(self):
        """
        获取所有情绪数据
        
        返回:
            dict: 包含所有情绪数据的字典
        """
        sentiment_data = {}
        
        # 获取恐慌贪婪指数
        fear_greed = self.fetch_fear_greed_index()
        if fear_greed is not None:
            sentiment_data['fear_greed_index'] = fear_greed
        
        # 获取社交媒体情绪
        social_sentiment = self.fetch_social_media_sentiment()
        if social_sentiment is not None:
            sentiment_data['social_media'] = social_sentiment
        
        # 获取Google趋势数据
        google_trends = self.fetch_google_trends()
        if google_trends is not None:
            sentiment_data['google_trends'] = google_trends
        
        # 获取Reddit情绪数据
        reddit_sentiment = self.fetch_reddit_sentiment()
        if reddit_sentiment is not None:
            sentiment_data['reddit'] = reddit_sentiment
        
        # 获取新闻情绪数据
        news_sentiment = self.analyze_news_sentiment()
        if news_sentiment is not None:
            sentiment_data['news'] = news_sentiment
        
        # 计算综合情绪指标
        if fear_greed is not None and google_trends is not None:
            # 简单的综合情绪计算示例
            fear_greed_value = fear_greed['value']
            google_trend_value = google_trends['currency_trend']
            composite_sentiment = (fear_greed_value + google_trend_value) / 2
            sentiment_data['composite_sentiment'] = composite_sentiment
        
        return sentiment_data
