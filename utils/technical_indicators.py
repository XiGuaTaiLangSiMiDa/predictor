import pandas as pd
import numpy as np
import ta

class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(close_prices, periods=14):
        """Calculate RSI with explicit handling of edge cases"""
        print("\nCalculating RSI...")
        
        # Calculate price changes
        delta = close_prices.diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate rolling averages
        avg_gain = gains.rolling(window=periods, min_periods=1).mean()
        avg_loss = losses.rolling(window=periods, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Handle division by zero and infinite values
        rsi = rsi.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with the median of non-NaN values
        median_rsi = rsi.median()
        rsi = rsi.fillna(median_rsi)
        
        # Clip values to 0-100 range
        rsi = np.clip(rsi, 0, 100)
        
        print(f"RSI range: [{rsi.min():.2f}, {rsi.max():.2f}]")
        print(f"RSI NaN values: {rsi.isna().sum()}")
        
        return rsi

    @staticmethod
    def add_all_indicators(df):
        """Add all technical indicators to the dataset"""
        print("\nAdding Technical Indicators...")
        df = df.copy()
        
        # Verify input data
        print(f"Input data shape: {df.shape}")
        print("Checking for NaN values in input data:")
        print(df.isna().sum())
        
        try:
            # Moving Averages
            print("\nCalculating Moving Averages...")
            df['ma20'] = ta.trend.SMAIndicator(df['close'], window=20, fillna=True).sma_indicator()
            df['ma50'] = ta.trend.SMAIndicator(df['close'], window=50, fillna=True).sma_indicator()
            df['ma200'] = ta.trend.SMAIndicator(df['close'], window=200, fillna=True).sma_indicator()
            
            # RSI
            df['rsi'] = TechnicalIndicators.calculate_rsi(df['close'])
            
            # MACD
            print("\nCalculating MACD...")
            macd = ta.trend.MACD(df['close'], fillna=True)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            print("\nCalculating Bollinger Bands...")
            bollinger = ta.volatility.BollingerBands(df['close'], fillna=True)
            df['bb_high'] = bollinger.bollinger_hband()
            df['bb_mid'] = bollinger.bollinger_mavg()
            df['bb_low'] = bollinger.bollinger_lband()
            
            # Volume Indicators
            print("\nCalculating Volume Indicators...")
            df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'], fillna=True).on_balance_volume()
            df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], fillna=True).chaikin_money_flow()
            
            # Volatility Indicators
            print("\nCalculating Volatility Indicators...")
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], fillna=True).average_true_range()
            df['volatility'] = df['close'].pct_change().rolling(window=20, min_periods=1).std() * np.sqrt(365)
            
            # Verify final data quality
            print("\nFinal data quality check:")
            print("NaN values after processing:")
            print(df.isna().sum())
            
            # Verify ranges
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    print(f"\n{col} range: [{df[col].min():.2f}, {df[col].max():.2f}]")
            
            return df
            
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            import traceback
            traceback.print_exc()
            raise

    @staticmethod
    def generate_trading_signals(df):
        """Generate trading signals based on technical indicators"""
        print("\nGenerating trading signals...")
        signals = pd.DataFrame(index=df.index)
        
        # Trend signals
        signals['ma_cross'] = np.where(df['ma20'] > df['ma50'], 1, -1)
        signals['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        
        # RSI signals
        signals['rsi_signal'] = np.where(df['rsi'] < 30, 1, np.where(df['rsi'] > 70, -1, 0))
        
        # Bollinger Bands signals
        signals['bb_signal'] = np.where(df['close'] < df['bb_low'], 1, 
                                      np.where(df['close'] > df['bb_high'], -1, 0))
        
        # Volume signals
        signals['volume_signal'] = np.where(df['volume'] > df['volume_sma'], 1, -1)
        
        # Composite signal
        signals['composite'] = (signals['ma_cross'] + signals['macd_cross'] + 
                              signals['rsi_signal'] + signals['bb_signal'] + 
                              signals['volume_signal']) / 5
        
        print("\nSignal statistics:")
        print(signals.describe())
        
        return signals

    @staticmethod
    def get_signal_interpretation(composite_signal):
        """Interpret the composite signal value"""
        if composite_signal > 0.6:
            return "Strong Buy"
        elif composite_signal > 0.2:
            return "Buy"
        elif composite_signal < -0.6:
            return "Strong Sell"
        elif composite_signal < -0.2:
            return "Sell"
        else:
            return "Neutral"

    @staticmethod
    def get_feature_columns():
        """Return the list of feature columns used for modeling"""
        return ['close', 'volume', 'rsi', 'macd', 'ma20', 'ma50', 'ma200',
                'bb_high', 'bb_low', 'obv', 'atr', 'volatility', 'cmf']
