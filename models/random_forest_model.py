import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from utils.data_cleaning import clean_dataframe

class RandomForestModel:
    def __init__(self, n_estimators=200, random_state=42, lookback=24):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.lookback = lookback
        self.feature_columns = None
    
    def prepare_data(self, df, features, target='close'):
        """Prepare data for Random Forest model"""
        # Clean the data
        df = clean_dataframe(df)
        self.feature_columns = features
        
        X, y = [], []
        for i in range(self.lookback, len(df)):
            X.append(df[features].iloc[i-self.lookback:i].values.flatten())
            y.append(df[target].iloc[i])
        
        return np.array(X), np.array(y)

    def train(self, X, y):
        """Train the Random Forest model"""
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """Generate predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)

    def predict_sequence(self, last_features, n_steps):
        """Generate sequence of predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = []
        current_features = last_features.copy()
        
        for _ in range(n_steps):
            next_pred = self.model.predict(current_features.reshape(1, -1))
            predictions.append(next_pred[0])
            
            # Update features for next prediction
            current_features = np.roll(current_features, -len(self.feature_columns))
            current_features[-len(self.feature_columns):] = next_pred
        
        return np.array(predictions)

    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get feature importance scores
        importance = self.model.feature_importances_
        
        # Create feature names for all lookback periods
        feature_names = []
        for i in range(self.lookback):
            for feature in self.feature_columns:
                feature_names.append(f"{feature}_t-{self.lookback-i}")
        
        # Create DataFrame with feature names and importance scores
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df

    def get_top_features(self, n=10):
        """Get top n most important features"""
        importance_df = self.get_feature_importance()
        return importance_df.head(n)

    def plot_feature_importance(self, n=20):
        """Plot feature importance"""
        import matplotlib.pyplot as plt
        
        importance_df = self.get_feature_importance()
        top_n = importance_df.head(n)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_n)), top_n['importance'])
        plt.xticks(range(len(top_n)), top_n['feature'], rotation=45, ha='right')
        plt.title(f'Top {n} Feature Importance')
        plt.tight_layout()
        plt.savefig('random_forest_feature_importance.png')
        plt.close()

    def get_model_params(self):
        """Get model parameters and statistics"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        params = {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'n_features': len(self.feature_columns) * self.lookback,
            'lookback_period': self.lookback
        }
        
        return params
