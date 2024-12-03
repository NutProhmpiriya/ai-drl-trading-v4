from datetime import datetime
from utils.data_processor import DataProcessor
from backtesting.backtest import BackTester
import os

def test_model(model_path: str, symbol: str = "USDJPY", timeframe: str = "5",
               start_date: datetime = datetime(2024, 1, 1),
               end_date: datetime = datetime(2024, 12, 31),
               initial_balance: float = 10000.0):
    """Test the trained model on unseen data"""
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Prepare test data
    data_processor = DataProcessor()
    df = data_processor.fetch_data(symbol, timeframe, start_date, end_date)
    
    if df is None or df.empty:
        print("Failed to fetch test data")
        return
    
    # Initialize backtester
    backtester = BackTester(model_path, initial_balance)
    
    # Run backtest
    trades_df, stats = backtester.run_backtest(df)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    trades_df.to_csv(f"results/trades_{symbol}_{timeframe}m_{timestamp}.csv")
    
    # Print statistics
    print("\nBacktest Results:")
    print("=" * 50)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2%}" if "Rate" in key or "Return" in key or "Drawdown" in key
                  else f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Plot results
    backtester.plot_results(df, trades_df, f"results/backtest_plot_{symbol}_{timeframe}m_{timestamp}.png")
    print(f"\nResults saved to results/trades_{symbol}_{timeframe}m_{timestamp}.csv")
    print(f"Plot saved to results/backtest_plot_{symbol}_{timeframe}m_{timestamp}.png")

if __name__ == "__main__":
    # Specify the path to your trained model
    model_path = "models/your_model_name.zip"  # Replace with your model path
    test_model(model_path)
