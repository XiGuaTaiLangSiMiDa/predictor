import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet
import ta
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class CryptoPredictor:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.symbol = 'GMT/USDT'
        self.scaler = MinMaxScaler()
        
    def fetch_historical_data(self):
        """Fetch historical price data from Binance"""
        try:
            timeframe = '1h'
            limit = 1000
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def add_technical_indicators(self, df):
        """Add technical indicators to the dataset"""
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['bb_high'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
        df['bb_low'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
        return df.dropna()

    def prepare_lstm_data(self, data, lookback=24):
        """Prepare data for LSTM model"""
        scaled_data = self.scaler.fit_transform(data[['close']])
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i])
        return np.array(X), np.array(y)

    def build_lstm_model(self, lookback):
        """Build LSTM model"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_prophet_model(self, df):
        """Train Facebook Prophet model"""
        prophet_df = df.reset_index()[['timestamp', 'close']].rename(
            columns={'timestamp': 'ds', 'close': 'y'})
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(prophet_df)
        return model

    def train_random_forest(self, df, features, target, lookback=24):
        """Train Random Forest model"""
        X, y = [], []
        for i in range(lookback, len(df)):
            X.append(df[features].iloc[i-lookback:i].values.flatten())
            y.append(df[target].iloc[i])
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

    def generate_predictions(self):
        """Generate predictions using ensemble of models"""
        df = self.fetch_historical_data()
        if df is None:
            return None
        
        df = self.add_technical_indicators(df)
        lookback = 24
        features = ['close', 'volume', 'rsi', 'macd', 'ema20']
        
        # Train models
        X_lstm, y_lstm = self.prepare_lstm_data(df, lookback)
        lstm_model = self.build_lstm_model(lookback)
        lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)
        
        prophet_model = self.train_prophet_model(df)
        rf_model = self.train_random_forest(df, features, 'close', lookback)
        
        # Generate predictions
        future_dates = pd.date_range(start=df.index[-1], periods=31, freq='D')[1:]
        predictions = pd.DataFrame(index=future_dates)
        
        # Prophet predictions
        prophet_future = prophet_model.make_future_dataframe(periods=30, freq='D')
        prophet_forecast = prophet_model.predict(prophet_future)
        predictions['prophet'] = prophet_forecast.tail(30)['yhat'].values
        
        # LSTM predictions
        last_sequence = X_lstm[-1]
        lstm_pred = []
        for _ in range(30):
            next_pred = lstm_model.predict(last_sequence.reshape(1, lookback, 1))
            lstm_pred.append(next_pred[0][0])
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = next_pred
        predictions['lstm'] = self.scaler.inverse_transform(np.array(lstm_pred).reshape(-1, 1))
        
        # Random Forest predictions
        last_features = df[features].tail(lookback).values.flatten()
        rf_pred = []
        for _ in range(30):
            next_pred = rf_model.predict(last_features.reshape(1, -1))
            rf_pred.append(next_pred[0])
            last_features = np.roll(last_features, -len(features))
            last_features[-len(features):] = next_pred
        predictions['random_forest'] = rf_pred
        
        # Ensemble prediction
        predictions['ensemble'] = (
            predictions['prophet'] * 0.3 + 
            predictions['lstm'] * 0.4 + 
            predictions['random_forest'] * 0.3
        )
        
        return predictions, df

    def plot_predictions(self, predictions, historical_data, plot_path):
        """Plot historical data and predictions"""
        plt.figure(figsize=(15, 8))
        
        plt.plot(historical_data.index[-60:], 
                historical_data['close'][-60:], 
                label='Historical', 
                color='blue')
        
        plt.plot(predictions.index, 
                predictions['ensemble'], 
                label='Prediction', 
                color='red', 
                linestyle='--')
        
        std = predictions.std(axis=1)
        plt.fill_between(predictions.index,
                        predictions['ensemble'] - std,
                        predictions['ensemble'] + std,
                        alpha=0.2,
                        color='red')
        
        plt.title(f'{self.symbol} Price Prediction (30 Days)', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USDT)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
