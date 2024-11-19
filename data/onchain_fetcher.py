import requests
from datetime import datetime
import ccxt

class OnchainFetcher:
    """链上数据获取类，处理所有区块链相关数据"""
    
    def __init__(self, base_currency='BTC', api_key=''):
        """
        初始化链上数据获取器
        
        参数:
            base_currency (str): 基础货币，如 'BTC'
            api_key (str): Glassnode API密钥
        """
        self.base_currency = base_currency
        self.glassnode_api_key = api_key
    
    def fetch_whale_holdings(self):
        """
        获取大户持仓数据
        
        返回:
            float: 大户持仓量
        """
        if not self.glassnode_api_key:
            print("Warning: Glassnode API key not set")
            return None
            
        try:
            url = f"https://api.glassnode.com/v1/metrics/distribution/balance_1pct_holders"
            params = {
                'api_key': self.glassnode_api_key,
                'asset': self.base_currency,
                'timestamp': int(datetime.now().timestamp())
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()[-1]['v']
            return None
        except Exception as e:
            print(f"Error fetching whale holdings: {e}")
            return None
    
    def fetch_exchange_flows(self):
        """
        获取交易所资金流向数据
        
        返回:
            dict: 包含流入流出数据的字典
        """
        try:
            exchanges = ['binance', 'coinbase', 'huobi']
            total_inflow = 0
            total_outflow = 0
            
            for exchange_id in exchanges:
                if hasattr(ccxt, exchange_id):
                    exchange = getattr(ccxt, exchange_id)()
                    try:
                        deposits = exchange.fetch_deposits(self.base_currency)
                        withdrawals = exchange.fetch_withdrawals(self.base_currency)
                        
                        total_inflow += sum(d['amount'] for d in deposits)
                        total_outflow += sum(w['amount'] for w in withdrawals)
                    except Exception as e:
                        print(f"Error fetching data from {exchange_id}: {e}")
            
            return {
                'inflow': total_inflow,
                'outflow': total_outflow,
                'net_flow': total_inflow - total_outflow
            }
        except Exception as e:
            print(f"Error fetching exchange flows: {e}")
            return None
    
    def fetch_active_addresses(self):
        """
        获取活跃地址数据
        
        返回:
            int: 活跃地址数量
        """
        if not self.glassnode_api_key:
            print("Warning: Glassnode API key not set")
            return None
            
        try:
            url = f"https://api.glassnode.com/v1/metrics/addresses/active_count"
            params = {
                'api_key': self.glassnode_api_key,
                'asset': self.base_currency,
                'timestamp': int(datetime.now().timestamp())
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()[-1]['v']
            return None
        except Exception as e:
            print(f"Error fetching active addresses: {e}")
            return None
    
    def fetch_miner_data(self):
        """
        获取矿工相关数据
        
        返回:
            dict: 包含矿工数据的字典
        """
        if not self.glassnode_api_key:
            print("Warning: Glassnode API key not set")
            return None
            
        try:
            miner_data = {}
            
            # 矿工余额
            url = f"https://api.glassnode.com/v1/metrics/mining/miner_balance"
            params = {
                'api_key': self.glassnode_api_key,
                'asset': self.base_currency,
                'timestamp': int(datetime.now().timestamp())
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                miner_data['balance'] = response.json()[-1]['v']
            
            # 矿工流出量
            url = f"https://api.glassnode.com/v1/metrics/mining/miner_outflow"
            response = requests.get(url, params=params)
            if response.status_code == 200:
                miner_data['outflow'] = response.json()[-1]['v']
            
            return miner_data
        except Exception as e:
            print(f"Error fetching miner data: {e}")
            return None
    
    def get_all_onchain_data(self):
        """
        获取所有链上数据
        
        返回:
            dict: 包含所有链上数据的字典
        """
        onchain_data = {}
        
        # 获取大户持仓数据
        whale_holdings = self.fetch_whale_holdings()
        if whale_holdings is not None:
            onchain_data['whale_holdings'] = whale_holdings
        
        # 获取交易所流向数据
        exchange_flows = self.fetch_exchange_flows()
        if exchange_flows is not None:
            onchain_data['exchange_flows'] = exchange_flows
        
        # 获取活跃地址数据
        active_addresses = self.fetch_active_addresses()
        if active_addresses is not None:
            onchain_data['active_addresses'] = active_addresses
        
        # 获取矿工数据
        miner_data = self.fetch_miner_data()
        if miner_data is not None:
            onchain_data['miner_data'] = miner_data
        
        return onchain_data
