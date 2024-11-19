import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

class LSTMModel:
    def __init__(self, lookback=24):
        self.lookback = lookback
        self.scaler = MinMaxScaler()
        self.model = None
        self.n_features = None
        self.feature_scaler = None
    
    def build_model(self, n_features):
        """Build LSTM model architecture"""
        print(f"\nBuilding LSTM model with {n_features} features...")
        self.n_features = n_features
        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=(self.lookback, n_features)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return model

    def prepare_data(self, data, feature_columns):
        """Prepare data for LSTM model"""
        print("\nPreparing LSTM data...")
        try:
            # Verify feature columns
            missing_columns = [col for col in feature_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")
            
            # Create feature scaler
            self.feature_scaler = MinMaxScaler()
            scaled_data = self.feature_scaler.fit_transform(data[feature_columns])
            
            print(f"Input shape: {scaled_data.shape}")
            
            X, y = [], []
            for i in range(self.lookback, len(scaled_data)):
                X.append(scaled_data[i-self.lookback:i])
                y.append(scaled_data[i][0])  # Predicting close price
            
            X = np.array(X)
            y = np.array(y)
            
            print(f"Prepared data shapes - X: {X.shape}, y: {y.shape}")
            return X, y, len(feature_columns)
            
        except Exception as e:
            print(f"Error preparing LSTM data: {e}")
            import traceback
            traceback.print_exc()
            raise

    def train(self, X, y, epochs=50, batch_size=32, verbose=0):
        """Train the LSTM model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        try:
            print(f"Training LSTM model with {len(X)} samples...")
            history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
            print("Training completed")
            return history
            
        except Exception as e:
            print(f"Error training LSTM model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def predict_sequence(self, last_sequence, n_steps):
        """Generate sequence of predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        try:
            print(f"\nGenerating {n_steps} LSTM predictions...")
            predictions = []
            curr_sequence = last_sequence.copy()
            
            # Verify input shape
            if curr_sequence.shape != (self.lookback, self.n_features):
                print(f"Reshaping input sequence from {curr_sequence.shape} to ({self.lookback}, {self.n_features})")
                curr_sequence = curr_sequence.reshape(self.lookback, self.n_features)
            
            for _ in range(n_steps):
                # Reshape for prediction
                input_seq = curr_sequence.reshape(1, self.lookback, self.n_features)
                
                # Get prediction
                next_pred = self.model.predict(input_seq, verbose=0)
                predictions.append(next_pred[0][0])
                
                # Update sequence
                curr_sequence = np.roll(curr_sequence, -1, axis=0)
                curr_sequence[-1] = next_pred[0]
            
            # Convert predictions to price values
            predictions = np.array(predictions).reshape(-1, 1)
            price_predictions = self.feature_scaler.inverse_transform(
                np.hstack([predictions, np.zeros((len(predictions), self.n_features-1))])
            )[:, 0]
            
            print(f"Generated predictions shape: {price_predictions.shape}")
            return price_predictions
            
        except Exception as e:
            print(f"Error generating LSTM predictions: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_feature_importance(self, X):
        """
        Estimate feature importance by measuring the variance of 
        LSTM activations for each feature
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        try:
            # Get the first LSTM layer's output for all sequences
            get_activations = Sequential([self.model.layers[0]])
            activations = get_activations.predict(X, verbose=0)
            
            # Calculate the variance of activations for each feature
            feature_importance = np.var(activations, axis=(0, 1))
            
            return feature_importance
            
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            return None
