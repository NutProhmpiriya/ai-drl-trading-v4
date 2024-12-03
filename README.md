# Forex Trading DRL Agent

Deep Reinforcement Learning (DRL) agent for Forex trading using PPO algorithm.

## Overview

This project implements a complete DRL-based trading system with three main components:

1. **Training System**:
   - Custom Gym Environment for Forex Trading
   - Data Processor for indicators and MT5 data
   - PPO Algorithm implementation
   - Model saving and management

2. **Testing System**:
   - Comprehensive backtesting
   - Performance metrics analysis
   - Trade visualization and history
   - CSV report generation

3. **Live Trading System**:
   - MT5 connection and real-time data streaming
   - Dynamic position sizing and risk management
   - Automatic trade execution with safety features
   - Performance monitoring

## Features

### Trading Strategy
- Currency Pair: USDJPY
- Timeframe: 5 Minutes
- Technical Indicators:
  - EMA (9, 21, 50)
  - RSI (14)
  - ATR (14)
  - OBV

### Risk Management
- Single position trading
- Dynamic SL/TP based on ATR
- Trail stop loss
- 1% risk per trade (auto position sizing)
- 1% max daily loss
- No weekend positions

### Performance Metrics
- Win Rate (separate for Buy/Sell)
- Profit Factor
- Sharpe Ratio
- Maximum Drawdown
- Total Return

## Installation

1. Make sure you have Python 3.8+ installed
2. Clone this repository
3. Run setup script:
```bash
setup.bat
```

4. Activate environment:
```bash
venv\Scripts\activate.bat
```

5. Verify installation:
```bash
python -c "import gymnasium; import stable_baselines3; import MetaTrader5; print('Setup successful!')"
```

## Project Structure
- `env/`: Custom Gym environment
- `models/`: Trained models
- `utils/`: Data processing and utilities
- `backtesting/`: Backtesting modules
- `live_trading/`: Live trading modules

## Usage

### Training
```bash
python train.py
```
Trains model on USDJPY 5M data from 2023

### Backtesting
```bash
python test_model.py --model models/your_model.zip
```
Tests model on USDJPY 5M data from 2024 and generates:
- Performance metrics
- Trade visualization
- Trade history CSV

### Live Trading
```bash
python live_trade.py --model models/your_model.zip [options]
```

Options:
- `--symbol`: Trading symbol (default: USDJPY)
- `--timeframe`: Minutes (default: 5)
- `--risk`: Risk per trade (default: 0.01)
- `--max-daily-loss`: Daily loss limit (default: 0.01)
- `--check-interval`: Seconds between checks (default: 0.1)

## Important Notes
- Always test in demo account first
- Ensure MT5 is properly configured
- Monitor system performance regularly
- Consider periodic model retraining
