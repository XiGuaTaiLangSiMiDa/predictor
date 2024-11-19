import argparse
from predictor import CryptoPredictor

def parse_arguments():
    parser = argparse.ArgumentParser(description='Cryptocurrency Price Prediction and Analysis')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                      help='Trading pair symbol (default: BTC/USDT)')
    parser.add_argument('--periods', type=int, default=30,
                      help='Number of periods to predict (default: 30)')
    parser.add_argument('--analysis-only', action='store_true',
                      help='Only perform technical analysis without predictions')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"\nCryptocurrency Analysis and Prediction Tool")
    print(f"=========================================")
    print(f"Analyzing {args.symbol}")
    
    try:
        # Initialize predictor
        predictor = CryptoPredictor(symbol=args.symbol)
        
        if args.analysis_only:
            # Fetch and analyze current market data
            df, signals = predictor.prepare_data()
            if df is not None:
                market_summary = predictor.data_fetcher.get_market_summary()
                print("\nMarket Summary:")
                for key, value in market_summary.items():
                    print(f"{key}: {value}")
                
                print("\nTechnical Analysis:")
                print_market_summary(df, signals)
        else:
            # Perform full analysis with predictions
            results = predictor.analyze_market()
            if results:
                # Get model insights
                insights = predictor.get_model_insights()
                if insights:
                    print("\nModel Insights:")
                    for model_name, importance in insights.items():
                        if importance is not None:
                            print(f"\n{model_name.upper()} Feature Importance:")
                            print(importance)
                
                print("\nAnalysis complete. Check prediction_plot.png for visualization.")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
