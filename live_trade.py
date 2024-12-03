from live_trading.trader import LiveTrader
import argparse

def main():
    parser = argparse.ArgumentParser(description='Live trading with DRL model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--symbol', type=str, default='USDJPY', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='5', help='Trading timeframe in minutes')
    parser.add_argument('--risk', type=float, default=0.01, help='Risk per trade (1% = 0.01)')
    parser.add_argument('--max-daily-loss', type=float, default=0.01, help='Maximum daily loss (1% = 0.01)')
    parser.add_argument('--check-interval', type=int, default=5, help='Interval to check for new trades (seconds)')
    
    args = parser.parse_args()
    
    try:
        trader = LiveTrader(
            model_path=args.model,
            symbol=args.symbol,
            timeframe=args.timeframe,
            risk_per_trade=args.risk,
            max_daily_loss=args.max_daily_loss
        )
        
        trader.run(check_interval=args.check_interval)
        
    except KeyboardInterrupt:
        print("\nStopping live trading...")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
