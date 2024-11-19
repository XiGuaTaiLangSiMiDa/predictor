import pandas as pd
import numpy as np
from utils.visualization import plot_predictions, plot_technical_indicators, print_market_summary

class MarketAnalyzer:
    """市场分析器类，处理所有市场分析相关的功能"""
    
    def __init__(self, data_fetcher, data_processor):
        """
        初始化市场分析器
        
        参数:
            data_fetcher (DataFetcher): 数据获取器实例
            data_processor (DataProcessor): 数据处理器实例
        """
        self.data_fetcher = data_fetcher
        self.data_processor = data_processor
    
    def analyze_market(self, predictions, df, signals):
        """
        执行市场分析
        
        参数:
            predictions (pd.DataFrame): 预测结果
            df (pd.DataFrame): 历史数据
            signals (pd.DataFrame): 交易信号
            
        返回:
            dict: 分析结果
        """
        try:
            # 获取市场数据
            market_summary = self.data_fetcher.get_market_summary()
            
            # 生成可视化
            self._generate_visualizations(predictions, df, signals)
            
            # 打印分析结果
            print_market_summary(df, signals)
            
            # 打印预测结果
            self._print_predictions(predictions)
            
            # 返回完整分析结果
            return {
                'predictions': predictions,
                'historical_data': df,
                'signals': signals,
                'market_summary': market_summary,
                'analysis': self._generate_analysis_report(df, signals, market_summary)
            }
            
        except Exception as e:
            print(f"Error in market analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_visualizations(self, predictions, df, signals):
        """
        生成可视化图表
        
        参数:
            predictions (pd.DataFrame): 预测结果
            df (pd.DataFrame): 历史数据
            signals (pd.DataFrame): 交易信号
        """
        try:
            plot_predictions(predictions, df, signals)
            plot_technical_indicators(df)
        except Exception as e:
            print(f"Error generating visualizations: {e}")
    
    def _print_predictions(self, predictions):
        """
        打印预测结果
        
        参数:
            predictions (pd.DataFrame): 预测结果
        """
        print("\nPrice Predictions for Next 30 Days:")
        for date, row in predictions.iterrows():
            print(f"{date.strftime('%Y-%m-%d')}: {row['ensemble']:.2f} USDT")
    
    def _generate_analysis_report(self, df, signals, market_summary):
        """
        生成分析报告
        
        参数:
            df (pd.DataFrame): 历史数据
            signals (pd.DataFrame): 交易信号
            market_summary (dict): 市场概况
            
        返回:
            dict: 分析报告
        """
        report = {
            'technical_analysis': self._analyze_technical_indicators(df),
            'market_sentiment': self._analyze_market_sentiment(market_summary),
            'trading_signals': self._analyze_trading_signals(signals),
            'risk_metrics': self._calculate_risk_metrics(df),
            'market_trends': self._analyze_market_trends(df)
        }
        return report
    
    def _analyze_technical_indicators(self, df):
        """
        分析技术指标
        
        参数:
            df (pd.DataFrame): 历史数据
            
        返回:
            dict: 技术分析结果
        """
        latest = df.iloc[-1]
        return {
            'rsi': {
                'value': latest['rsi'],
                'status': 'Overbought' if latest['rsi'] > 70 else 'Oversold' if latest['rsi'] < 30 else 'Neutral'
            },
            'macd': {
                'value': latest['macd'],
                'signal': latest['macd_signal'],
                'histogram': latest['macd_diff'],
                'trend': 'Bullish' if latest['macd'] > latest['macd_signal'] else 'Bearish'
            },
            'moving_averages': {
                'ma20': latest['ma20'],
                'ma50': latest['ma50'],
                'ma200': latest['ma200'],
                'trend': self._analyze_ma_trend(latest)
            },
            'bollinger_bands': {
                'upper': latest['bb_high'],
                'middle': latest['bb_mid'],
                'lower': latest['bb_low'],
                'status': self._analyze_bb_status(latest)
            }
        }
    
    def _analyze_market_sentiment(self, market_summary):
        """
        分析市场情绪
        
        参数:
            market_summary (dict): 市场概况
            
        返回:
            dict: 市场情绪分析结果
        """
        sentiment = {}
        
        if market_summary and 'sentiment' in market_summary:
            sentiment_data = market_summary['sentiment']
            if 'fear_greed_index' in sentiment_data:
                fear_greed = sentiment_data['fear_greed_index']
                if isinstance(fear_greed, dict):
                    sentiment['fear_greed'] = {
                        'value': fear_greed.get('value'),
                        'classification': fear_greed.get('classification')
                    }
        
        return sentiment
    
    def _analyze_trading_signals(self, signals):
        """
        分析交易信号
        
        参数:
            signals (pd.DataFrame): 交易信号
            
        返回:
            dict: 交易信号分析结果
        """
        latest = signals.iloc[-1]
        return {
            'ma_cross': 'Bullish' if latest['ma_cross'] > 0 else 'Bearish',
            'macd_cross': 'Bullish' if latest['macd_cross'] > 0 else 'Bearish',
            'rsi_signal': self._interpret_rsi_signal(latest['rsi_signal']),
            'bb_signal': self._interpret_bb_signal(latest['bb_signal']),
            'volume_signal': 'Increasing' if latest['volume_signal'] > 0 else 'Decreasing',
            'composite': {
                'value': latest['composite'],
                'interpretation': self._interpret_composite_signal(latest['composite'])
            }
        }
    
    def _calculate_risk_metrics(self, df):
        """
        计算风险指标
        
        参数:
            df (pd.DataFrame): 历史数据
            
        返回:
            dict: 风险指标
        """
        returns = df['close'].pct_change()
        return {
            'volatility': returns.std() * np.sqrt(365),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(df['close']),
            'var_95': returns.quantile(0.05),
            'var_99': returns.quantile(0.01)
        }
    
    def _analyze_market_trends(self, df):
        """
        分析市场趋势
        
        参数:
            df (pd.DataFrame): 历史数据
            
        返回:
            dict: 市场趋势分析结果
        """
        return {
            'short_term': self._analyze_trend(df, window=20),
            'medium_term': self._analyze_trend(df, window=50),
            'long_term': self._analyze_trend(df, window=200)
        }
    
    def _analyze_ma_trend(self, latest):
        """分析移动平均趋势"""
        if latest['ma20'] > latest['ma50'] > latest['ma200']:
            return 'Strong Uptrend'
        elif latest['ma20'] < latest['ma50'] < latest['ma200']:
            return 'Strong Downtrend'
        elif latest['ma20'] > latest['ma50']:
            return 'Short-term Uptrend'
        else:
            return 'Short-term Downtrend'
    
    def _analyze_bb_status(self, latest):
        """分析布林带状态"""
        if latest['close'] > latest['bb_high']:
            return 'Overbought'
        elif latest['close'] < latest['bb_low']:
            return 'Oversold'
        else:
            return 'Normal'
    
    def _interpret_rsi_signal(self, signal):
        """解释RSI信号"""
        if signal > 0:
            return 'Buy'
        elif signal < 0:
            return 'Sell'
        return 'Neutral'
    
    def _interpret_bb_signal(self, signal):
        """解释布林带信号"""
        if signal > 0:
            return 'Buy'
        elif signal < 0:
            return 'Sell'
        return 'Neutral'
    
    def _interpret_composite_signal(self, composite):
        """解释综合信号"""
        if composite > 0.6:
            return 'Strong Buy'
        elif composite > 0.2:
            return 'Buy'
        elif composite < -0.6:
            return 'Strong Sell'
        elif composite < -0.2:
            return 'Sell'
        return 'Neutral'
    
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """计算夏普比率"""
        excess_returns = returns - risk_free_rate/365
        return np.sqrt(365) * excess_returns.mean() / returns.std()
    
    def _calculate_max_drawdown(self, prices):
        """计算最大回撤"""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def _analyze_trend(self, df, window):
        """分析特定时间窗口的趋势"""
        ma = df['close'].rolling(window=window).mean()
        current_price = df['close'].iloc[-1]
        ma_current = ma.iloc[-1]
        
        if current_price > ma_current:
            strength = (current_price / ma_current - 1) * 100
            return {'direction': 'Uptrend', 'strength': strength}
        else:
            strength = (ma_current / current_price - 1) * 100
            return {'direction': 'Downtrend', 'strength': strength}
