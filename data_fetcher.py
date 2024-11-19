import ccxt
import pandas as pd
import requests
from utils.data_cleaning import clean_dataframe, validate_data

class DataFetcher:
    def __init__(self, spot_symbol='BTC/USDT', futures_symbol='BTC/USDT:USDT'):
        self.exchange = ccxt.binance()
        self.spot_symbol = spot_symbol
        self.futures_symbol = futures_symbol
    
    def fetch_historical_data(self, timeframe='1h', limit=1000):
        """Fetch historical price data from Binance"""
        try:
            print(f"\nFetching historical data for {self.spot_symbol}...")
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(self.spot_symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Validate the data
            validate_data(df, ['close', 'volume'])
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None

    def fetch_market_data(self):
        """Fetch additional market data including funding rate, open interest, and sentiment"""
        try:
            market_data = {}
            
            # Fetch Funding Rate
            try:
                funding_rate = self.exchange.fetch_funding_rate(self.futures_symbol)
                market_data['funding_rate'] = funding_rate['fundingRate'] if funding_rate else None
                print(f"Funding Rate: {market_data['funding_rate']}")
            except Exception as e:
                print(f"Error fetching funding rate: {e}")
                market_data['funding_rate'] = None
            
            # Fetch Open Interest
            try:
                ticker = self.exchange.fetch_ticker(self.spot_symbol)
                market_data['open_interest'] = ticker['info'].get('openInterest', None)
                print(f"Open Interest: {market_data['open_interest']}")
            except Exception as e:
                print(f"Error fetching open interest: {e}")
                market_data['open_interest'] = None
            
            # Fetch Fear and Greed Index
            try:
                fear_greed_response = requests.get('https://api.alternative.me/fng/')
                if fear_greed_response.status_code == 200:
                    fear_greed_data = fear_greed_response.json()
                    market_data['fear_greed_index'] = fear_greed_data['data'][0]['value']
                    print(f"Fear & Greed Index: {market_data['fear_greed_index']}")
                else:
                    market_data['fear_greed_index'] = None
            except Exception as e:
                print(f"Error fetching fear and greed index: {e}")
                market_data['fear_greed_index'] = None
            
            return market_data
            
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None

    def fetch_order_book(self, limit=20):
        """Fetch current order book"""
        try:
            order_book = self.exchange.fetch_order_book(self.spot_symbol, limit)
            return {
                'bids': order_book['bids'],
                'asks': order_book['asks']
            }
        except Exception as e:
            print(f"Error fetching order book: {e}")
            return None

    def fetch_recent_trades(self, limit=100):
        """Fetch recent trades"""
        try:
            trades = self.exchange.fetch_trades(self.spot_symbol, limit=limit)
            trades_df = pd.DataFrame(trades)
            return trades_df
        except Exception as e:
            print(f"Error fetching recent trades: {e}")
            return None

    def fetch_exchange_info(self):
        """Fetch exchange information for the symbol"""
        try:
            markets = self.exchange.load_markets()
            return markets.get(self.spot_symbol)
        except Exception as e:
            print(f"Error fetching exchange info: {e}")
            return None

    def get_market_summary(self):
        """Get a comprehensive market summary"""
        summary = {}
        
        # Get ticker information
        try:
            ticker = self.exchange.fetch_ticker(self.spot_symbol)
            summary.update({
                'last_price': ticker['last'],
                '24h_volume': ticker['quoteVolume'],
                '24h_change': ticker['percentage'],
                '24h_high': ticker['high'],
                '24h_low': ticker['low']
            })
        except Exception as e:
            print(f"Error fetching ticker: {e}")
        
        # Get market data
        market_data = self.fetch_market_data()
        if market_data:
            summary.update(market_data)
        
        return summary

    def validate_symbol(self):
        """Validate if the trading pair exists and is active"""
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
