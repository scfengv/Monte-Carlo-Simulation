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
    
    print("\n========== Portfolio Analysis Results ==========")
    print("Risk Metrics:")
    for metric, value in analysis["Risk_Metrics"].items():
        if metric not in ["Sharpe_Ratio", "Sortino_Ratio"]:
            print(f"{metric}: {value:.4f}%")
        else:
            print(f"{metric}: {value:.4f}")
    
    print("\nSimulation Results:")
    print(f"Mean Final Value: ${analysis['Simulation_Results']['Mean']:,.2f}")
    print(f"Median Final Value: ${analysis['Simulation_Results']['Median']:,.2f}")
    print(f"Standard Deviation: {(analysis['Simulation_Results']['Std'] / analysis['Simulation_Results']['Mean']):,.4f}")
    
    print("\nConfidence Intervals:")
    for level, (lower, upper) in analysis["Simulation_Results"]["Confidence_Intervals"].items():
        print(f"{level} Confidence Interval: ${lower:,.2f} to ${upper:,.2f}")

if __name__ == "__main__":
    main()