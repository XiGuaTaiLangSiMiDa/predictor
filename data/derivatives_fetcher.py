import ccxt
import requests

class DerivativesFetcher:
    """衍生品市场数据获取类，处理所有衍生品相关数据"""
    
    def __init__(self, spot_symbol='BTC/USDT', futures_symbol='BTC/USDT:USDT'):
        """
        初始化衍生品数据获取器
        
        参数:
            spot_symbol (str): 现货交易对
            futures_symbol (str): 合约交易对
        """
        self.exchange = ccxt.binance()
        self.spot_symbol = spot_symbol
        self.futures_symbol = futures_symbol
        self.base_currency = spot_symbol.split('/')[0]  # 'BTC'
    
    def fetch_funding_rate(self):
        """
        获取资金费率
        
        返回:
            float: 资金费率
        """
        try:
            funding_rate = self.exchange.fetch_funding_rate(self.futures_symbol)
            return funding_rate['fundingRate'] if funding_rate else None
        except Exception as e:
            print(f"Error fetching funding rate: {e}")
            return None
    
    def fetch_premium_rate(self):
        """
        获取期货溢价率
        
        返回:
            float: 期货溢价率（百分比）
        """
        try:
            spot_ticker = self.exchange.fetch_ticker(self.spot_symbol)
            futures_ticker = self.exchange.fetch_ticker(self.futures_symbol)
            
            if spot_ticker and futures_ticker:
                spot_price = spot_ticker['last']
                futures_price = futures_ticker['last']
                premium_rate = (futures_price - spot_price) / spot_price * 100
                return premium_rate
            return None
        except Exception as e:
            print(f"Error calculating premium rate: {e}")
            return None
    
    def fetch_open_interest(self):
        """
        获取未平仓合约量
        
        返回:
            float: 未平仓合约量
        """
        try:
            ticker = self.exchange.fetch_ticker(self.futures_symbol)
            return ticker['info'].get('openInterest', None)
        except Exception as e:
            print(f"Error fetching open interest: {e}")
            return None
    
    def fetch_long_short_ratio(self):
        """
        获取多空持仓比
        
        返回:
            float: 多空持仓比
        """
        try:
            url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
            params = {
                'symbol': f"{self.base_currency}USDT",
                'period': '5m'
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data:
                    return float(data[0]['longShortRatio'])
            return None
        except Exception as e:
            print(f"Error fetching long/short ratio: {e}")
            return None
    
    def fetch_liquidation_data(self):
        """
        获取清算数据
        
        返回:
            dict: 包含清算数据的字典
        """
        try:
            url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
            params = {
                'symbol': f"{self.base_currency}USDT",
                'period': '5m'
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data:
                    return {
                        'long_ratio': float(data[0]['longAccount']),
                        'short_ratio': float(data[0]['shortAccount']),
                        'long_short_ratio': float(data[0]['longShortRatio'])
                    }
            return None
        except Exception as e:
            print(f"Error fetching liquidation data: {e}")
            return None
    
    def fetch_futures_volume(self):
        """
        获取期货成交量数据
        
        返回:
            dict: 包含成交量数据的字典
        """
        try:
            ticker = self.exchange.fetch_ticker(self.futures_symbol)
            return {
                'volume': ticker['quoteVolume'],
                'volume_usd': ticker['quoteVolume'] * ticker['last']
            }
        except Exception as e:
            print(f"Error fetching futures volume: {e}")
            return None
    
    def get_all_derivatives_data(self):
        """
        获取所有衍生品市场数据
        
        返回:
            dict: 包含所有衍生品市场数据的字典
        """
        derivatives_data = {}
        
        # 获取资金费率
        funding_rate = self.fetch_funding_rate()
        if funding_rate is not None:
            derivatives_data['funding_rate'] = funding_rate
        
        # 获取期货溢价率
        premium_rate = self.fetch_premium_rate()
        if premium_rate is not None:
            derivatives_data['premium_rate'] = premium_rate
        
        # 获取未平仓合约量
        open_interest = self.fetch_open_interest()
        if open_interest is not None:
            derivatives_data['open_interest'] = open_interest
        
        # 获取多空持仓比
        long_short_ratio = self.fetch_long_short_ratio()
        if long_short_ratio is not None:
            derivatives_data['long_short_ratio'] = long_short_ratio
        
        # 获取清算数据
        liquidation_data = self.fetch_liquidation_data()
        if liquidation_data is not None:
            derivatives_data['liquidation_data'] = liquidation_data
        
        # 获取期货成交量数据
        futures_volume = self.fetch_futures_volume()
        if futures_volume is not None:
            derivatives_data['futures_volume'] = futures_volume
        
        return derivatives_data
