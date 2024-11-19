from prophet import Prophet
import pandas as pd
import numpy as np
from utils.data_cleaning import clean_dataframe

class ProphetModel:
    def __init__(self, daily_seasonality=True, yearly_seasonality=True):
        self.model = None
        self.daily_seasonality = daily_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.regressors = ['rsi', 'volume', 'macd']
    
    def prepare_data(self, df):
        """Prepare data for Prophet model"""
        print("\nPreparing Prophet data...")
        
        try:
            # Create a copy of the dataframe
            df_copy = df.copy()
            
            # Create prophet dataframe with timestamp and target variable
            prophet_df = pd.DataFrame()
            prophet_df['ds'] = df_copy.index
            prophet_df['y'] = df_copy['close'].values
            
            # Verify target variable
            if prophet_df['y'].isna().any():
                print("Warning: NaN values found in target variable 'y', filling with forward fill")
                prophet_df['y'] = prophet_df['y'].ffill().bfill()
            
            print(f"Target variable range: [{prophet_df['y'].min():.2f}, {prophet_df['y'].max():.2f}]")
            
            # Add regressors
            for regressor in self.regressors:
                if regressor in df_copy.columns:
                    # Get values and handle NaN/inf
                    values = df_copy[regressor].values
                    values = np.where(np.isinf(values), np.nan, values)
                    
                    # Calculate median excluding NaN values
                    median_value = np.nanmedian(values)
                    if np.isnan(median_value):
                        print(f"Warning: All values are NaN for {regressor}, using 0")
                        median_value = 0
                    
                    # Fill NaN values with median
                    values = np.where(np.isnan(values), median_value, values)
                    
                    # Add to prophet dataframe
                    prophet_df[regressor] = values
                    
                    print(f"Regressor {regressor}:")
                    print(f"  Range: [{np.min(values):.2f}, {np.max(values):.2f}]")
                    print(f"  Median: {median_value:.2f}")
                    print(f"  NaN values after filling: {np.isnan(values).sum()}")
                else:
                    print(f"Warning: Regressor {regressor} not found in data, using zeros")
                    prophet_df[regressor] = 0
            
            # Final verification
            print("\nFinal Prophet dataframe shape:", prophet_df.shape)
            print("Columns:", prophet_df.columns.tolist())
            
            # Check for any remaining NaN values
            nan_counts = prophet_df.isna().sum()
            if nan_counts.any():
                print("\nWarning: Remaining NaN values:")
                print(nan_counts[nan_counts > 0])
                # Final attempt to clean any remaining NaN values
                prophet_df = prophet_df.ffill().bfill()
            
            print("Final NaN check:", prophet_df.isna().sum().sum() == 0)
            
            return prophet_df
            
        except Exception as e:
            print(f"Error in prepare_data: {e}")
            import traceback
            traceback.print_exc()
            raise

    def train(self, prophet_df):
        """Train the Prophet model"""
        try:
            # Verify data before training
            nan_check = prophet_df.isna().sum()
            if nan_check.any():
                print("\nError: NaN values found in columns:")
                print(nan_check[nan_check > 0])
                raise ValueError("Prophet data contains NaN values")
            
            # Create and configure the model
            self.model = Prophet(
                daily_seasonality=self.daily_seasonality,
                yearly_seasonality=self.yearly_seasonality,
                uncertainty_samples=1000
            )
            
            # Add regressors
            for regressor in self.regressors:
                if regressor in prophet_df.columns:
                    print(f"Adding regressor: {regressor}")
                    self.model.add_regressor(regressor)
            
            # Fit the model
            print("Training Prophet model...")
            self.model.fit(prophet_df)
            print("Prophet model training completed")
            return self.model
            
        except Exception as e:
            print(f"Error in train: {e}")
            import traceback
            traceback.print_exc()
            raise

    def predict(self, periods=30, freq='D', prophet_df=None):
        """Generate predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
            
            # Add regressor values for future dates
            if prophet_df is not None:
                for regressor in self.regressors:
                    if regressor in prophet_df.columns:
                        # Calculate the median value for each regressor
                        median_value = prophet_df[regressor].median()
                        # Use median value for future predictions
                        future[regressor] = median_value
            
            # Make predictions
            forecast = self.model.predict(future)
            return forecast
            
        except Exception as e:
            print(f"Error in predict: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_components(self):
        """Get the trend, seasonal, and holiday components of the forecast"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.component_modes

    def plot_components(self, forecast):
        """Plot the components of the forecast"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        self.model.plot_components(forecast)

    def get_feature_importance(self):
        """Get the importance of each regressor"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        try:
            # Extract regressor coefficients
            params = self.model.params
            regressor_coeffs = {}
            
            if 'beta' in params:
                for i, name in enumerate(self.regressors):
                    if i < len(params['beta']):
                        regressor_coeffs[name] = params['beta'][i]
            
            # Convert to DataFrame for easier analysis
            importance_df = pd.DataFrame({
                'regressor': list(regressor_coeffs.keys()),
                'coefficient': list(regressor_coeffs.values())
            })
            if not importance_df.empty:
                importance_df['abs_coefficient'] = abs(importance_df['coefficient'])
                importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
            
            return importance_df
            
        except Exception as e:
            print(f"Error in get_feature_importance: {e}")
            return pd.DataFrame()
