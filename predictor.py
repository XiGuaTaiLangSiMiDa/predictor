import pandas as pd
import numpy as np
from data_fetcher import DataFetcher
from technical_indicators import TechnicalIndicators
from models.lstm_model import LSTMModel
from models.prophet_model import ProphetModel
from models.random_forest_model import RandomForestModel
from utils.visualization import plot_predictions, plot_technical_indicators, print_market_summary
from utils.data_cleaning import clean_dataframe, validate_data, check_data_quality, print_data_quality_report

class CryptoPredictor:
    def __init__(self, symbol='BTC/USDT'):
        self.spot_symbol = symbol
        self.futures_symbol = f"{symbol}:USDT"
        self.data_fetcher = DataFetcher(self.spot_symbol, self.futures_symbol)
        self.lstm_model = LSTMModel()
        self.prophet_model = ProphetModel()
        self.rf_model = RandomForestModel()
        self.lookback = 24
        
    def prepare_data(self):
        """Fetch and prepare data for prediction"""
        print("\nPreparing data for prediction...")
        
        try:
            # Fetch historical data
            df = self.data_fetcher.fetch_historical_data()
            if df is None:
                raise ValueError("Failed to fetch historical data")
            
            # Check initial data quality
            initial_quality = check_data_quality(df)
            print("\nInitial data quality report:")
            print_data_quality_report(initial_quality)
            
            # Add technical indicators
            df = TechnicalIndicators.add_all_indicators(df)
            
            # Generate trading signals
            signals = TechnicalIndicators.generate_trading_signals(df)
            df = pd.concat([df, signals], axis=1)
            
            # Clean the combined dataset
            df = clean_dataframe(df)
            
            # Check final data quality
            final_quality = check_data_quality(df)
            print("\nFinal data quality report:")
            print_data_quality_report(final_quality)
            
            return df, signals
            
        except Exception as e:
            print(f"Error in prepare_data: {e}")
            import traceback
            traceback.print_exc()
            raise

    def train_models(self, df):
        """Train all models"""
        print("\nTraining models...")
        
        try:
            feature_columns = TechnicalIndicators.get_feature_columns()
            
            # Train LSTM
            print("\nTraining LSTM model...")
            X_lstm, y_lstm, n_features = self.lstm_model.prepare_data(df, feature_columns)
            self.lstm_model.build_model(n_features)
            self.lstm_model.train(X_lstm, y_lstm)
            print("LSTM training completed")
            
            # Train Prophet
            print("\nTraining Prophet model...")
            prophet_df = self.prophet_model.prepare_data(df)
            self.prophet_model.train(prophet_df)
            print("Prophet training completed")
            
            # Train Random Forest
            print("\nTraining Random Forest model...")
            X_rf, y_rf = self.rf_model.prepare_data(df, feature_columns)
            self.rf_model.train(X_rf, y_rf)
            print("Random Forest training completed")
            
            return X_lstm, prophet_df
            
        except Exception as e:
            print(f"Error in train_models: {e}")
            import traceback
            traceback.print_exc()
            raise

    def generate_predictions(self, periods=30):
        """Generate predictions using all models"""
        try:
            # Prepare data
            df, signals = self.prepare_data()
            X_lstm, prophet_df = self.train_models(df)
            
            # Fetch market data for adjustments
            market_data = self.data_fetcher.fetch_market_data()
            
            print("\nGenerating predictions...")
            future_dates = pd.date_range(start=df.index[-1], periods=periods+1, freq='D')[1:]
            predictions = pd.DataFrame(index=future_dates)
            model_predictions = {}
            
            # Generate predictions from each model
            try:
                # Prophet predictions
                prophet_forecast = self.prophet_model.predict(periods=periods, prophet_df=prophet_df)
                model_predictions['prophet'] = prophet_forecast.tail(periods)['yhat'].values
                print("Prophet predictions generated")
            except Exception as e:
                print(f"Error generating Prophet predictions: {e}")
                model_predictions['prophet'] = None
            
            try:
                # LSTM predictions
                lstm_preds = self.lstm_model.predict_sequence(X_lstm[-1], periods)
                model_predictions['lstm'] = lstm_preds.flatten()[:periods]  # Ensure correct length
                print("LSTM predictions generated")
            except Exception as e:
                print(f"Error generating LSTM predictions: {e}")
                model_predictions['lstm'] = None
            
            try:
                # Random Forest predictions
                feature_columns = TechnicalIndicators.get_feature_columns()
                last_features = df[feature_columns].tail(self.lookback).values.flatten()
                rf_preds = self.rf_model.predict_sequence(last_features, periods)
                model_predictions['random_forest'] = rf_preds
                print("Random Forest predictions generated")
            except Exception as e:
                print(f"Error generating Random Forest predictions: {e}")
                model_predictions['random_forest'] = None
            
            # Add successful predictions to DataFrame
            for model, preds in model_predictions.items():
                if preds is not None:
                    predictions[model] = preds
            
            # Check if we have any valid predictions
            if predictions.empty:
                raise ValueError("No valid predictions generated from any model")
            
            # Calculate ensemble prediction
            weights = {
                'prophet': 0.3,
                'lstm': 0.4,
                'random_forest': 0.3
            }
            
            # Initialize ensemble column with zeros
            predictions['ensemble'] = 0.0
            total_weight = 0.0
            
            # Add weighted predictions from each successful model
            for model, weight in weights.items():
                if model in predictions.columns:
                    predictions['ensemble'] += predictions[model] * weight
                    total_weight += weight
            
            # Normalize by total weight if not all models succeeded
            if total_weight > 0 and total_weight != 1.0:
                predictions['ensemble'] = predictions['ensemble'] / total_weight
            
            # Adjust predictions based on market data
            if market_data:
                fear_greed_factor = float(market_data['fear_greed_index']) / 100 if market_data['fear_greed_index'] else 0.5
                funding_rate_factor = 1 + (market_data['funding_rate'] if market_data['funding_rate'] else 0)
                predictions['ensemble'] = predictions['ensemble'] * fear_greed_factor * funding_rate_factor
            
            print("\nPrediction generation completed")
            return predictions, df, signals
            
        except Exception as e:
            print(f"Error generating predictions: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def analyze_market(self):
        """Perform comprehensive market analysis"""
        try:
            # Get current market data
            market_summary = self.data_fetcher.get_market_summary()
            
            # Generate predictions
            predictions, df, signals = self.generate_predictions()
            if predictions is None:
                return
            
            # Generate visualizations
            plot_predictions(predictions, df, signals)
            plot_technical_indicators(df)
            
            # Print analysis
            print_market_summary(df, signals)
            
            # Print predictions
            print("\nPrice Predictions for Next 30 Days:")
            for date, row in predictions.iterrows():
                print(f"{date.strftime('%Y-%m-%d')}: {row['ensemble']:.2f} USDT")
            
            # Return all data for further analysis if needed
            return {
                'predictions': predictions,
                'historical_data': df,
                'signals': signals,
                'market_summary': market_summary
            }
            
        except Exception as e:
            print(f"Error in market analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_model_insights(self):
        """Get insights about feature importance from all models"""
        insights = {}
        
        try:
            if hasattr(self, 'X_lstm'):
                insights['lstm'] = self.lstm_model.get_feature_importance(self.X_lstm)
        except Exception as e:
            print(f"Error getting LSTM insights: {e}")
        
        try:
            if self.prophet_model.model:
                insights['prophet'] = self.prophet_model.get_feature_importance()
        except Exception as e:
            print(f"Error getting Prophet insights: {e}")
        
        try:
            if self.rf_model.model:
                insights['random_forest'] = self.rf_model.get_feature_importance()
        except Exception as e:
            print(f"Error getting Random Forest insights: {e}")
        
        return insights
