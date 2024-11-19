import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

class LSTMModel:
    def __init__(self, lookback=24):
        """
        初始化LSTM模型
        
        参数:
            lookback (int): 用于预测的历史数据点数量
        """
        self.lookback = lookback
        self.model = None
        self.n_features = None
        self.feature_scaler = MinMaxScaler()  # 初始化特征缩放器
    
    def build_model(self, n_features):
        """
        构建LSTM模型架构
        
        参数:
            n_features (int): 特征数量
        """
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
        """
        准备LSTM模型的训练数据
        
        参数:
            data (pd.DataFrame): 输入数据
            feature_columns (list): 特征列名列表
            
        返回:
            tuple: (X, y, n_features)
        """
        print("\nPreparing LSTM data...")
        try:
            # 验证特征列
            missing_columns = [col for col in feature_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")
            
            # 缩放特征
            scaled_data = self.feature_scaler.fit_transform(data[feature_columns])
            
            print(f"Input shape: {scaled_data.shape}")
            
            X, y = [], []
            for i in range(self.lookback, len(scaled_data)):
                X.append(scaled_data[i-self.lookback:i])
                y.append(scaled_data[i][0])  # 预测收盘价
            
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
        """
        训练LSTM模型
        
        参数:
            X (np.array): 训练特征
            y (np.array): 训练目标
            epochs (int): 训练轮数
            batch_size (int): 批次大小
            verbose (int): 输出详细程度
        """
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
        """
        生成预测序列
        
        参数:
            last_sequence (np.array): 最后的输入序列
            n_steps (int): 预测步数
            
        返回:
            np.array: 预测序列
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        try:
            print(f"\nGenerating {n_steps} LSTM predictions...")
            predictions = []
            curr_sequence = last_sequence.copy()
            
            # 验证输入形状
            if curr_sequence.shape != (self.lookback, self.n_features):
                print(f"Reshaping input sequence from {curr_sequence.shape} to ({self.lookback}, {self.n_features})")
                curr_sequence = curr_sequence.reshape(self.lookback, self.n_features)
            
            for _ in range(n_steps):
                # 重塑用于预测的序列
                input_seq = curr_sequence.reshape(1, self.lookback, self.n_features)
                
                # 获取预测
                next_pred = self.model.predict(input_seq, verbose=0)
                predictions.append(next_pred[0][0])
                
                # 更新序列
                curr_sequence = np.roll(curr_sequence, -1, axis=0)
                curr_sequence[-1] = next_pred[0]
            
            # 转换预测为价格值
            predictions = np.array(predictions).reshape(-1, 1)
            # 创建完整的特征数组（用0填充其他特征）
            full_features = np.zeros((len(predictions), self.n_features))
            full_features[:, 0] = predictions.flatten()
            
            # 反向转换预测值
            price_predictions = self.feature_scaler.inverse_transform(full_features)[:, 0]
            
            print(f"Generated predictions shape: {price_predictions.shape}")
            return price_predictions
            
        except Exception as e:
            print(f"Error generating LSTM predictions: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_feature_importance(self, X):
        """
        估计特征重要性
        
        参数:
            X (np.array): 输入数据
            
        返回:
            np.array: 特征重要性分数
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        try:
            # 获取第一个LSTM层对所有序列的输出
            get_activations = Sequential([self.model.layers[0]])
            activations = get_activations.predict(X, verbose=0)
            
            # 计算每个特征的激活方差
            feature_importance = np.var(activations, axis=(0, 1))
            
            return feature_importance
            
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            return None
