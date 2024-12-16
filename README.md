# Monte Carlo Simulation Model
Build: Dec 14, 2024

## Introduction
This Monte Carlo Simulation Model combines Modern Portfolio Theory (MPT) with Monte Carlo simulation (MC) to provide risk assessment and portfolio optimization. This model features optimal distribution fitting, dynamic risk assessment, and efficient frontier analysis to help make informed investment decisions.

## Methods
This project implements an advanced portfolio analysis system with several key features:

- **Monte Carlo Simulation**: Generates thousands (adjustable) of possible future scenarios using historical data and optimal distribution fitting
- **Portfolio Optimization**: Implements modern portfolio theory and SLSQP algorithm to optimize asset allocation based on various objectives (Sharpe ratio, minimum risk, maximum return)
- **Distribution Analysis**: Automatically determines the best statistical distribution for each asset's returns (Normal distribution, GausianHMM, Gausian Mixture Model, Kernel Distribution Estimation, etc.)
- **Risk Metrics**: Calculates comprehensive risk metrics including VaR, CVaR, and maximum drawdown
- **Dynamic Risk Assessment**: Uses EWMA for dynamic covariance estimation
- **Transaction Costs**: Incorporates transaction costs into portfolio optimization
- **Efficient Frontier**: Generates and visualizes the efficient frontier
- **Individual Stock Analysis**: Provides detailed analysis of individual stock characteristics

## Requirements
- Python 3.x

```bash
pip install -r requirements.txt
```

## Usage

1. Basic usage with default parameters:

```python
# run.py
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
analysis, stock_metrics, efficient_frontier = run_portfolio_analysis(
    tickers = tickers,
    initial_investment = 100000
)
```

2. Advanced usage with custom parameters:

```python
# run.py
analysis, stock_metrics, efficient_frontier = run_portfolio_analysis(
    tickers = tickers,
    initial_investment = 100000,
    num_simulations = 1000,
    time_horizon = 252,  # One trading year
    optimization_goal = "sharpe",  # Options: "sharpe", "min_risk", "max_return"
    min_weight = 0,     # Minimum weight per asset
    max_weight = 0.4    # Maximum weight per asset
    transaction_costs = np.array([0.001] * len(tickers))  # 0.1% transaction cost per asset
)
```

## Disclaimer
- This tool is for educational and research purposes only. It should not be considered as financial advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.