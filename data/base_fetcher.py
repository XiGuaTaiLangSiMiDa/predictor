import ccxt
import pandas as pd
from datetime import datetime
from utils.data_cleaning import clean_dataframe, validate_data

class BaseFetcher:
    """基础数据获取类，处理基本的市场数据"""
    
    def __init__(self, spot_symbol='BTC/USDT', futures_symbol='BTC/USDT:USDT'):
        """
        初始化数据获取器
        
        参数:
            spot_symbol (str): 现货交易对
            futures_symbol (str): 合约交易对
        """
        self.exchange = ccxt.binance()
        self.spot_symbol = spot_symbol
        self.futures_symbol = futures_symbol
        self.base_currency = spot_symbol.split('/')[0]  # 'BTC'
    
    def fetch_historical_data(self, timeframe='1h', limit=1000):
        """
        获取历史价格数据
        
        参数:
            timeframe (str): 时间周期，如 '1h', '4h', '1d'
            limit (int): 获取的数据点数量
            
        返回:
            pandas.DataFrame: 包含OHLCV数据的DataFrame
        """
        try:
            print(f"\nFetching historical data for {self.spot_symbol}...")
            
            ohlcv = self.exchange.fetch_ohlcv(self.spot_symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            validate_data(df, ['close', 'volume'])
            return df
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None
    
    def fetch_ticker(self):
        """
        获取当前市场行情数据
        
        返回:
            dict: 包含当前市场数据的字典
        """
        try:
            ticker = self.exchange.fetch_ticker(self.spot_symbol)
            return {
                'last_price': ticker['last'],
                '24h_volume': ticker['quoteVolume'],
                '24h_change': ticker['percentage'],
                '24h_high': ticker['high'],
                '24h_low': ticker['low']
            }
        except Exception as e:
            print(f"Error fetching ticker: {e}")
            return None
    
    def fetch_order_book(self, limit=20):
        """
        获取当前订单簿数据
        
        参数:
            limit (int): 获取的订单数量
            
        返回:
            dict: 包含买卖盘数据的字典
        """
        try:
            order_book = self.exchange.fetch_order_book(self.spot_symbol, limit)
            return {
                'bids': order_book['bids'],  # 买盘
                'asks': order_book['asks']   # 卖盘
            }
        except Exception as e:
            print(f"Error fetching order book: {e}")
            return None
    
    def fetch_recent_trades(self, limit=100):
        """
        获取最近成交记录
        
        参数:
            limit (int): 获取的成交记录数量
            
        返回:
            pandas.DataFrame: 包含成交记录的DataFrame
        """
        try:
            trades = self.exchange.fetch_trades(self.spot_symbol, limit=limit)
            trades_df = pd.DataFrame(trades)
            return trades_df
        except Exception as e:
            print(f"Error fetching recent trades: {e}")
            return None
    
    def validate_symbol(self):
        """
        验证交易对是否有效
        
        返回:
            bool: 交易对是否有效
        """
        try:
            markets = self.exchange.load_markets()
            if self.spot_symbol not in markets:
                raise ValueError(f"Symbol {self.spot_symbol} not found in exchange")
            
            market = markets[self.spot_symbol]
            if not market['active']:
                raise ValueError(f"Symbol {self.spot_symbol} is not active")
            
            return True
        except Exception as e:
            print(f"Error validating symbol: {e}")
            return False
    
    def get_exchange_info(self):
        """
        获取交易所信息
        
        返回:
            dict: 交易所信息
        """
        try:
            markets = self.exchange.load_markets()
            return markets.get(self.spot_symbol)
        except Exception as e:
            print(f"Error fetching exchange info: {e}")
            return None
    
    def get_market_summary(self):
        """
        获取市场概况
        
        返回:
            dict: 市场概况数据
        """
        summary = {}
        
        # 获取基本市场数据
        ticker_data = self.fetch_ticker()
        if ticker_data:
            summary.update(ticker_data)
        
        # 获取订单簿深度
        order_book = self.fetch_order_book()
        if order_book:
            summary['order_book'] = {
                'bid_count': len(order_book['bids']),
                'ask_count': len(order_book['asks']),
                'top_bid': order_book['bids'][0][0] if order_book['bids'] else None,
                'top_ask': order_book['asks'][0][0] if order_book['asks'] else None
            }
        
        return summary
