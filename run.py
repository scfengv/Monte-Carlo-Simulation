from mc import *

def main():
    tickers = ["TSLA", "AAPL", "AMD", "AMZN", "AVGO", "META", "PLTR", "NVDA", "SMR", "TSM", "ON", "JD", "BABA"]
    
    analysis, stock_metrics, efficient_frontier = run_portfolio_analysis(
        tickers = tickers,
        initial_investment = 100000,
        num_simulations = 1000,
        time_horizon = 30,
        optimization_goal = "sharpe",
        min_weight = 0,    # Minimum 0% in each stock
        max_weight = 0.4   # Maximum 40% in each stock
    )
    
    print("\nBasic Statistics:")
    print(analysis['Basic Statistics'])

    print("\nRisk Metrics:")
    print(analysis['Risk Metrics'])

    print("\nConfidence Intervals:")
    print(analysis['Confidence Intervals'])

    print("\nDistribution Information:")
    print(analysis['Distribution Information'])

if __name__ == "__main__":
    main()