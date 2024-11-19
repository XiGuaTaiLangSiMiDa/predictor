import argparse
from predictor import CryptoPredictor

def parse_arguments():
    parser = argparse.ArgumentParser(description='加密货币价格预测和分析工具')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                      help='交易对符号 (默认: BTC/USDT)')
    parser.add_argument('--periods', type=int, default=30,
                      help='预测天数 (默认: 30)')
    parser.add_argument('--analysis-only', action='store_true',
                      help='仅执行技术分析，不进行预测')
    parser.add_argument('--use-onchain', action='store_true',
                      help='使用链上数据进行分析')
    parser.add_argument('--use-derivatives', action='store_true',
                      help='使用衍生品市场数据进行分析')
    parser.add_argument('--use-sentiment', action='store_true',
                      help='使用市场情绪数据进行分析')
    return parser.parse_args()

def print_market_data_summary(market_summary):
    """打印市场数据摘要"""
    print("\n市场数据摘要")
    print("============")
    
    if 'onchain' in market_summary:
        print("\n链上数据:")
        onchain = market_summary['onchain']
        if 'whale_holdings' in onchain:
            print(f"大户持仓量: {onchain['whale_holdings']:.2f} BTC")
        if 'exchange_inflow' in onchain:
            print(f"交易所流入量: {onchain['exchange_inflow']:.2f} BTC")
        if 'exchange_outflow' in onchain:
            print(f"交易所流出量: {onchain['exchange_outflow']:.2f} BTC")
        if 'net_flow' in onchain:
            print(f"净流入量: {onchain['net_flow']:.2f} BTC")
        if 'active_addresses' in onchain:
            print(f"活跃地址数: {onchain['active_addresses']}")
        if 'miner_balance' in onchain:
            print(f"矿工余额: {onchain['miner_balance']:.2f} BTC")
        if 'miner_outflow' in onchain:
            print(f"矿工流出量: {onchain['miner_outflow']:.2f} BTC")
    
    if 'derivatives' in market_summary:
        print("\n衍生品市场数据:")
        derivatives = market_summary['derivatives']
        if 'funding_rate' in derivatives:
            print(f"资金费率: {derivatives['funding_rate']*100:.4f}%")
        if 'premium_rate' in derivatives:
            print(f"期货溢价率: {derivatives['premium_rate']:.2f}%")
        if 'open_interest' in derivatives:
            print(f"未平仓合约量: {derivatives['open_interest']:.2f}")
        if 'long_short_ratio' in derivatives:
            print(f"多空持仓比: {derivatives['long_short_ratio']:.2f}")
    
    if 'sentiment' in market_summary:
        print("\n市场情绪数据:")
        sentiment = market_summary['sentiment']
        if 'fear_greed_index' in sentiment:
            print(f"恐慌贪婪指数: {sentiment['fear_greed_index']}")
        if 'google_trends' in sentiment:
            print(f"谷歌趋势指数: {sentiment['google_trends']}")

def print_trading_signals(signals):
    """打印交易信号分析"""
    if signals is None:
        return
    
    print("\n交易信号分析")
    print("============")
    
    latest_signals = signals.iloc[-1]
    
    # MA交叉信号
    ma_signal = "多头" if latest_signals['ma_cross'] > 0 else "空头"
    print(f"\nMA交叉信号: {ma_signal}")
    print(f"MA20: {latest_signals.get('ma20', 'N/A'):.2f}")
    print(f"MA50: {latest_signals.get('ma50', 'N/A'):.2f}")
    
    # MACD信号
    macd_signal = "多头" if latest_signals['macd_cross'] > 0 else "空头"
    print(f"\nMACD信号: {macd_signal}")
    print(f"MACD: {latest_signals.get('macd', 'N/A'):.2f}")
    print(f"MACD Signal: {latest_signals.get('macd_signal', 'N/A'):.2f}")
    
    # RSI信号
    rsi = latest_signals.get('rsi', 0)
    rsi_signal = "超买" if rsi > 70 else "超卖" if rsi < 30 else "中性"
    print(f"\nRSI信号: {rsi_signal}")
    print(f"RSI值: {rsi:.2f}")
    
    # 布林带信号
    bb_signal = latest_signals.get('bb_signal', 0)
    bb_interpretation = "超买" if bb_signal < 0 else "超卖" if bb_signal > 0 else "中性"
    print(f"\n布林带信号: {bb_interpretation}")
    
    # 成交量信号
    volume_signal = "放量" if latest_signals['volume_signal'] > 0 else "缩量"
    print(f"\n成交量信号: {volume_signal}")
    
    # 综合信号
    composite = latest_signals['composite']
    if composite > 0.6:
        interpretation = "强烈买入"
    elif composite > 0.2:
        interpretation = "买入"
    elif composite < -0.6:
        interpretation = "强烈卖出"
    elif composite < -0.2:
        interpretation = "卖出"
    else:
        interpretation = "中性"
    
    print(f"\n综合信号: {interpretation}")
    print(f"信号强度: {composite:.2f}")

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    print(f"\n加密货币分析和预测工具")
    print(f"====================")
    print(f"分析对象: {args.symbol}")
    
    try:
        # 初始化预测器
        predictor = CryptoPredictor(symbol=args.symbol)
        
        if args.analysis_only:
            # 获取并分析当前市场数据
            df, signals = predictor.prepare_data()
            if df is not None:
                market_summary = predictor.data_fetcher.get_market_summary()
                
                # 打印市场数据摘要
                if args.use_onchain or args.use_derivatives or args.use_sentiment:
                    print_market_data_summary(market_summary)
                
                # 打印技术分析结果
                print_trading_signals(signals)
        else:
            # 执行完整分析和预测
            results = predictor.analyze_market()
            if results:
                # 打印市场数据摘要
                if args.use_onchain or args.use_derivatives or args.use_sentiment:
                    print_market_data_summary(results['market_summary'])
                
                # 打印交易信号
                print_trading_signals(results['signals'])
                
                # 获取模型洞察
                insights = predictor.get_model_insights()
                if insights:
                    print("\n模型特征重要性")
                    print("==============")
                    for model_name, importance in insights.items():
                        if importance is not None:
                            print(f"\n{model_name.upper()} 模型特征重要性:")
                            print(importance)
                
                print("\n分析完成。查看 prediction_plot.png 获取可视化结果。")
    
    except KeyboardInterrupt:
        print("\n操作被用户取消。")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
