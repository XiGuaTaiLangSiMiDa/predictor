import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def plot_predictions(predictions, historical_data, signals, save_path='prediction_plot.png'):
    """Plot enhanced visualization with technical indicators and signals"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price and predictions
    ax1.plot(historical_data.index[-60:], historical_data['close'][-60:], 
            label='Historical', color='blue')
    ax1.plot(predictions.index, predictions['ensemble'], 
            label='Prediction', color='red', linestyle='--')
    
    # Add Bollinger Bands
    ax1.plot(historical_data.index[-60:], historical_data['bb_high'][-60:], 
            'g--', alpha=0.5, label='BB High')
    ax1.plot(historical_data.index[-60:], historical_data['bb_low'][-60:], 
            'g--', alpha=0.5, label='BB Low')
    
    # Add confidence interval
    std = predictions.std(axis=1)
    ax1.fill_between(predictions.index,
                    predictions['ensemble'] - std,
                    predictions['ensemble'] + std,
                    alpha=0.2, color='red')
    
    # Plot signals
    ax2.plot(signals.index[-60:], signals['composite'][-60:], 
            label='Trading Signal', color='purple')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Formatting
    ax1.set_title('BTC/USDT Price Prediction with Technical Indicators', fontsize=14)
    ax1.set_ylabel('Price (USDT)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    
    ax2.set_title('Trading Signals', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Signal Strength', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_technical_indicators(df, save_path='technical_indicators.png'):
    """Plot key technical indicators for analysis"""
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))
    
    # Price and Moving Averages
    axes[0].plot(df.index, df['close'], label='Price', color='blue')
    axes[0].plot(df.index, df['ma20'], label='MA20', color='red')
    axes[0].plot(df.index, df['ma50'], label='MA50', color='green')
    axes[0].plot(df.index, df['ma200'], label='MA200', color='purple')
    axes[0].set_title('Price and Moving Averages')
    axes[0].legend()
    axes[0].grid(True)
    
    # RSI
    axes[1].plot(df.index, df['rsi'], color='orange')
    axes[1].axhline(y=70, color='r', linestyle='--')
    axes[1].axhline(y=30, color='g', linestyle='--')
    axes[1].set_title('RSI')
    axes[1].grid(True)
    
    # MACD
    axes[2].plot(df.index, df['macd'], label='MACD', color='blue')
    axes[2].plot(df.index, df['macd_signal'], label='Signal', color='red')
    axes[2].bar(df.index, df['macd_diff'], label='MACD Histogram', color='gray', alpha=0.3)
    axes[2].set_title('MACD')
    axes[2].legend()
    axes[2].grid(True)
    
    # Volume and OBV
    ax_vol = axes[3]
    ax_obv = ax_vol.twinx()
    ax_vol.bar(df.index, df['volume'], label='Volume', color='gray', alpha=0.3)
    ax_obv.plot(df.index, df['obv'], label='OBV', color='green')
    ax_vol.set_title('Volume and OBV')
    ax_vol.legend(loc='upper left')
    ax_obv.legend(loc='upper right')
    ax_vol.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def print_market_summary(df, signals):
    """Print a summary of current market conditions and signals"""
    print("\nBTC/USDT Analysis Summary")
    print("=========================")
    
    # Current market conditions
    print("\nCurrent Market Conditions:")
    print(f"RSI: {df['rsi'].iloc[-1]:.2f}")
    print(f"MACD: {df['macd'].iloc[-1]:.2f}")
    print(f"Current Volatility: {df['volatility'].iloc[-1]:.2f}%")
    print(f"ATR: {df['atr'].iloc[-1]:.2f}")
    
    # Trading signals
    print("\nTrading Signals (Last Value):")
    print(f"Composite Signal: {signals['composite'].iloc[-1]:.2f}")
    signal_interpretation = "Strong Buy" if signals['composite'].iloc[-1] > 0.6 else \
                          "Buy" if signals['composite'].iloc[-1] > 0.2 else \
                          "Strong Sell" if signals['composite'].iloc[-1] < -0.6 else \
                          "Sell" if signals['composite'].iloc[-1] < -0.2 else \
                          "Neutral"
    print(f"Signal Interpretation: {signal_interpretation}")
