import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, timedelta
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis, ks_2samp, gaussian_kde, skewnorm, t as student_t

warnings.filterwarnings('ignore')

class OptimalDistributionFinder:
    """
    Find the optimal distribution for each stock's returns
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.best_method = None
        self.fitted_models = {}
    
    def fit_distributions(self, returns):
        # Normal distribution
        self.fitted_models['normal'] = {
            'mean': returns.mean(),
            'std': returns.std()
        }
        
        # Skewed normal distribution
        self.fitted_models['skewed_normal'] = {
            'mean': returns.mean(),
            'std': returns.std(),
            'skewness': skew(returns)
        }
        
        # Add Student's t-distribution
        df, loc, scale = student_t.fit(returns)
        self.fitted_models['student_t'] = {
            'df': df,
            'loc': loc,
            'scale': scale
        }
        
        # Gaussian Mixture Model
        returns_array = np.array(returns).reshape(-1, 1)
        best_bic = np.inf
        best_gmm = None
        for n in range(1, 6):
            gmm = GaussianMixture(n_components = n, random_state = 2024)
            gmm.fit(returns_array)
            bic = gmm.bic(returns_array)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        self.fitted_models['gmm'] = best_gmm
        
        # Regime Switching Model
        self.fitted_models['regime_switching'] = self.fit_regime_switching(returns)
        
        # Kernel Density Estimation
        self.fitted_models['kde'] = gaussian_kde(returns)

    def fit_regime_switching(self, returns):
        returns_array = returns.values if isinstance(returns, pd.Series) else returns
        data = returns_array.reshape(-1, 1)
        hmm = GaussianHMM(n_components = 2, random_state = 2024)
        hmm.fit(data)
        return hmm
    
    def generate_samples(self, method, size):
        if method == 'normal':
            params = self.fitted_models['normal']
            return np.random.normal(params['mean'], params['std'], size)

        elif method == 'skewed_normal':
            params = self.fitted_models['skewed_normal']
            return skewnorm.rvs(params['skewness'], loc = params['mean'], scale = params['std'], size = size)

        elif method == 'student_t':
            params = self.fitted_models['student_t']
            return student_t.rvs(df = params['df'], loc = params['loc'], 
                                 scale = params['scale'], size = size)
            
        elif method == 'gmm':
            return self.fitted_models['gmm'].sample(size)[0].flatten()

        elif method == 'regime_switching':
            hmm = self.fitted_models['regime_switching']
            states, _ = hmm.sample(size)
            return states.flatten()
        
        elif method == 'kde':
            return self.fitted_models['kde'].resample(size)[0]
    
    def evaluate_fit(self, returns, method):
        """
        Kolmogorov-Smirnov Test (KS) & Moment Comparisons

        ### KS ###
        - Maximum distance between the empirical cumulative distribution functions of two samples
        - Advantages: Distribution-free

        ### Moment ###
        - Compares four statistical moments between the fitted distribution and actual returns
            Mean error:               Central tendency
            Standard deviation error: Spread
            Skewness error:           Asymmetry
            Kurtosis error:           Tail behavior
        """

        samples = self.generate_samples(method, len(returns))
        ks_stat, _ = ks_2samp(returns, samples)
        moment_errors = {
            'mean_error': abs(np.mean(samples) - np.mean(returns)),
            'std_error': abs(np.std(samples) - np.std(returns)),
            'skew_error': abs(skew(samples) - skew(returns)),
            'kurt_error': abs(kurtosis(samples) - kurtosis(returns))
        }
        
        # Lower score indicates better fit
        return ks_stat + sum(moment_errors.values())
    
    def find_best_distribution(self, returns):
        self.fit_distributions(returns)

        for method in ['normal', 'skewed_normal', 'student_t', 'gmm', 'regime_switching', 'kde']:
            scores = {method: self.evaluate_fit(returns, method)}

        self.best_method = min(scores.items(), key = lambda x: x[1])[0]
        return self.best_method, scores

class PortfolioOptimizer:
    """
    Optimize portfolio weights considering 
    1. transaction costs
    2. different objectives
    3. dynamic risk
    """
    def __init__(self, returns, transaction_costs, risk_free_rate = 0.02):
        self.returns = returns
        self.transaction_costs = transaction_costs
        self.risk_free_rate = risk_free_rate
        self.num_assets = returns.shape[1]

    def calculate_dynamic_covariance(self, returns, lambda_decay = 0.94):
        """
        EWMA for dynamic covariance estimation
        """
        weights = np.array([(1-lambda_decay) * lambda_decay**i 
                           for i in range(len(returns))])
        weights = weights[::-1] / weights.sum()
        weighted_returns = returns * weights[:, np.newaxis]
        return weighted_returns.T @ weighted_returns
         
    def calculate_portfolio_metrics(self, weights):
        """
        1. Portfolio return
        2. Volatility
        3. Sharpe ratio

        How to calculate the Portfolio volatility (σp)
        - Annualized standard deviation of portfolio returns

        σp = √(∑ᵢ∑ⱼ wᵢwⱼσᵢⱼ x 252)
        where:
            wᵢ = weight of asset i
            wⱼ = weight of asset j
            σᵢⱼ = covariance between assets i and j
        """
        dynamic_cov = self.calculate_dynamic_covariance(self.returns)
        port_return = np.sum(self.returns.mean() * weights) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(dynamic_cov * 252, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol
        return port_return, port_vol, sharpe
    
    def optimize(self, objective = "sharpe", target_return = None, target_risk = None, 
                current_weights = None, min_weight = 0.0, max_weight = 1.0):
        """
        Optimize portfolio weights based on different objectives
        
        Parameters:
        - objective: 'sharpe', 'min_risk', 'max_return', or 'efficient_frontier'
        - target_return: Target portfolio return (for efficient frontier)
        - target_risk: Target portfolio risk (for efficient frontier)
        - current_weights: Current portfolio weights (for transaction cost calculation)
        - min_weight: Minimum weight constraint
        - max_weight: Maximum weight constraint
        """
        if current_weights is None:
            current_weights = np.zeros(self.num_assets)
        
        def objective_function(weights):
            port_return, port_vol, sharpe = self.calculate_portfolio_metrics(weights)
            transaction_cost = np.sum(np.abs(weights - current_weights) * self.transaction_costs)
            adjusted_return = port_return - transaction_cost
            
            if objective == "sharpe":
                return -sharpe
            elif objective == "min_risk":
                return port_vol
            elif objective == "max_return":
                return -adjusted_return
            elif objective == "efficient_frontier":
                return port_vol
        
        # Constraints
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        if target_return is not None:
            constraints.append({
                "type": "eq",
                "fun": lambda x: self.calculate_portfolio_metrics(x)[0] - target_return
            })
        
        if target_risk is not None:
            constraints.append({
                "type": "eq",
                "fun": lambda x: self.calculate_portfolio_metrics(x)[1] - target_risk
            })
        
        # Bounds for weights
        bounds = tuple((min_weight, max_weight) for _ in range(self.num_assets))
        
        # Initial equal weights
        initial_weights = np.array([1/self.num_assets] * self.num_assets)
        
        # Optimize using Sequential Least Squares Programming (SLSQP)
        result = minimize(objective_function, initial_weights, method = "SLSQP",
                          bounds = bounds, constraints = constraints
        )
        return result.x
    
    def generate_efficient_frontier(self, num_portfolios = 50):
        # Find minimum risk and maximum return portfolios
        min_risk_weights = self.optimize(objective = "min_risk")
        max_return_weights = self.optimize(objective = "max_return")
        
        min_ret, min_vol, _ = self.calculate_portfolio_metrics(min_risk_weights)
        max_ret, max_vol, _ = self.calculate_portfolio_metrics(max_return_weights)
        
        # Generate portfolios along the efficient frontier
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        efficient_portfolios = []
        
        for target_ret in target_returns:
            weights = self.optimize(objective = "efficient_frontier", target_return = target_ret)
            ret, vol, sharpe = self.calculate_portfolio_metrics(weights)
            efficient_portfolios.append({
                'return': ret,
                'volatility': vol,
                'sharpe': sharpe,
                'weights': weights
            })
            
        return pd.DataFrame(efficient_portfolios)

class StockAnalyzer:
    """
    Analyze individual stock characteristics and contributions
    """
    def __init__(self, returns, prices):
        self.returns = returns
        self.prices = prices
        self.benchmark = None
        self.load_benchmark()
    
    def load_benchmark(self):
        """Load and prepare benchmark data"""
        spy = yf.Ticker("SPY")
        spy_prices = spy.history(
            start = self.prices.index[0],
            end = self.prices.index[-1]
        )["Close"]
        self.benchmark = spy_prices.pct_change().dropna()
    
    def calculate_metrics(self):
        metrics = {}
        
        for ticker in self.returns.columns:
            stock_returns = self.returns[ticker]
            stock_prices = self.prices[ticker]
            
            # Basic statistics
            annual_return = stock_returns.mean() * 252
            annual_vol = stock_returns.std() * np.sqrt(252)
            
            # Risk metrics
            var_95 = np.percentile(stock_returns, 5)
            cvar_95 = stock_returns[stock_returns <= var_95].mean()
            
            # Maximum drawdown
            cum_returns = (1 + stock_returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = cum_returns / rolling_max - 1
            max_drawdown = float(drawdowns.min())
            
            # Risk-adjusted returns
            sharpe = annual_return / annual_vol
            sortino = annual_return / (stock_returns[stock_returns < 0].std() * np.sqrt(252))

            # Beta and Information ratio
            beta, info_ratio = self.calculate_metrics_with_benchmark(stock_returns)
            
            # Distribution characteristics
            metrics[ticker] = {
                'Annual Return': annual_return,
                'Annual Volatility': annual_vol,
                'Sharpe Ratio': sharpe,
                'Sortino Ratio': sortino,
                'VaR (95%)': -var_95 * 100,
                'CVaR (95%)': -cvar_95 * 100,
                'Max Drawdown': max_drawdown * 100,
                'Skewness': skew(stock_returns),
                'Kurtosis': kurtosis(stock_returns),
                'Beta': beta,
                'Information Ratio': info_ratio
            }
        
        return pd.DataFrame(metrics).T

    def calculate_metrics_with_benchmark(self, stock_returns):
        """
        Calculate beta and IR relative to market benchmark
        """
        aligned_data = pd.concat([stock_returns, self.benchmark], axis = 1).dropna()
        stock_returns_aligned = aligned_data.iloc[:, 0]
        benchmark_returns_aligned = aligned_data.iloc[:, 1]
        
        # Beta calculation
        covariance = np.cov(stock_returns_aligned, benchmark_returns_aligned)[0][1]
        market_variance = np.var(benchmark_returns_aligned)
        beta = covariance / market_variance if market_variance != 0 else 1
        
        # Information Ratio calculation
        active_returns = stock_returns_aligned - benchmark_returns_aligned
        ir = (
            active_returns.mean() / active_returns.std() 
            if active_returns.std() != 0 else 0
        ) * np.sqrt(252)
        
        return beta, ir
    
    def plot_metrics(self, metrics_df):
        fig = plt.figure(figsize = (20, 15))
        
        # 1. Risk-Return Scatter Plot
        ax1 = plt.subplot(221)
        ax1.scatter(metrics_df['Annual Volatility'], metrics_df['Annual Return'])
        for idx in metrics_df.index:
            ax1.annotate(idx, (metrics_df.loc[idx, 'Annual Volatility'], 
                             metrics_df.loc[idx, 'Annual Return']))
        ax1.set_xlabel('Annual Volatility')
        ax1.set_ylabel('Annual Return')
        ax1.set_title('Risk-Return Profile')
        ax1.grid(True)
        
        # 2. Risk Metrics Comparison
        ax2 = plt.subplot(222)
        metrics_df[['VaR (95%)', 'CVaR (95%)', 'Max Drawdown']].plot(kind = 'bar', ax = ax2)
        ax2.set_title('Risk Metrics Comparison')
        ax2.set_ylabel('Value')
        plt.xticks(rotation = 45)
        
        # 3. Distribution Characteristics
        ax3 = plt.subplot(223)
        metrics_df[['Skewness', 'Kurtosis']].plot(kind = 'bar', ax = ax3)
        ax3.set_title('Distribution Characteristics')
        ax3.set_ylabel('Value')
        plt.xticks(rotation = 45)
        
        # 4. Risk-Adjusted Returns
        ax4 = plt.subplot(224)
        metrics_df[['Sharpe Ratio', 'Sortino Ratio', 'Information Ratio']].plot(kind = 'bar', ax = ax4)
        ax4.set_title('Risk-Adjusted Return Metrics')
        ax4.set_ylabel('Ratio')
        plt.xticks(rotation = 45)
        
        plt.tight_layout()
        plt.show()

class PortfolioAnalyzer:
    """
    Analyze portfolio performance and risk metrics
    """
    def __init__(self, tickers, weights, initial_investment, transaction_costs = None):
        self.tickers = tickers
        self.weights = weights
        self.initial_investment = initial_investment
        if transaction_costs is None:
            self.transaction_costs = np.array([0.001] * len(tickers))
        else:
            self.transaction_costs = transaction_costs
        
    def get_stock_data(self, years = 5):
        end_date = datetime.now()
        start_date = end_date - timedelta(days = years * 365)
        data = pd.DataFrame()
        for ticker in self.tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(start = start_date, end = end_date)["Close"]
            data[ticker] = hist

        return data
    
    def calculate_risk_metrics(self, returns):
        """
        Calculate portfolio risk metrics
        """
        portfolio_returns = np.sum(returns * self.weights, axis = 1)
        
        var_95 = -np.percentile(portfolio_returns, 5)
        cvar_95 = -portfolio_returns[portfolio_returns <= -var_95].mean()
        
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        cum_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        max_drawdown = float((cum_returns / rolling_max - 1).min())
        
        metrics = {
            "VaR_95": var_95 * 100,
            "CVaR_95": cvar_95 * 100,
            "Sharpe_Ratio": (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
            "Sortino_Ratio": (portfolio_returns.mean() * 252) / downside_std if downside_std > 0 else np.inf,
            "Max_Drawdown": max_drawdown * 100,
            "Downside_Volatility": downside_std * 100
        }
        return metrics
    
    def run_monte_carlo(self, num_simulations, time_horizon):
        """
        Run Monte Carlo simulation
        """
        stock_data = self.get_stock_data()
        returns = stock_data.pct_change().dropna()
        cov_matrix = returns.cov()
        
        # Find optimal distribution for each stock
        distribution_finders = {}
        for ticker in self.tickers:
            finder = OptimalDistributionFinder()
            finder.find_best_distribution(returns[ticker])
            distribution_finders[ticker] = finder
        
        # Run simulations
        simulation_results = np.zeros((num_simulations, time_horizon))
        for sim in range(num_simulations):
            uncorrelated_returns = np.array([
                distribution_finders[ticker].generate_samples(finder.best_method, time_horizon)
                for ticker, finder in distribution_finders.items()
            ]).T
            
            L = np.linalg.cholesky(cov_matrix)
            daily_returns = np.dot(uncorrelated_returns, L.T)
            portfolio_returns = np.sum(daily_returns * self.weights, axis = 1)
            simulation_results[sim] = np.cumprod(1 + portfolio_returns) * self.initial_investment
        
        return simulation_results, returns, distribution_finders
    
    def analyze_results(self, simulation_results, returns, distribution_finders):
        """
        Analyze simulation results
        """
        final_values = simulation_results[:, -1]
        confidence_levels = [0.9, 0.95, 0.99]
        
        analysis = {
            "Distribution_Info": {ticker: finder.best_method 
                                for ticker, finder in distribution_finders.items()},
            "Risk_Metrics": self.calculate_risk_metrics(returns),
            "Simulation_Results": {
                "Mean": np.mean(final_values),
                "Median": np.median(final_values),
                "Std": np.std(final_values),
                "Confidence_Intervals": {
                    f"{level:.0%}": (np.percentile(final_values, (1 - level) * 100),
                                     np.percentile(final_values, level * 100))
                    for level in confidence_levels
                }
            }
        }
        
        if self.transaction_costs is not None:
            total_cost = np.sum(np.abs(self.weights) * self.transaction_costs) * self.initial_investment
            analysis["Transaction_Costs"] = total_cost
        
        return analysis
    
    def plot_results(self, simulation_results):
        """
        Plot simulation results
        """
        plt.figure(figsize = (12, 8))
        for i in range(len(simulation_results)):
            plt.plot(simulation_results[i], "b", alpha = 0.1)
        
        percentiles = np.percentile(simulation_results, [5, 50, 95], axis = 0)
        plt.plot(percentiles[1], "r", linewidth = 2, label = "Median")
        plt.fill_between(range(len(percentiles[0])), percentiles[0], percentiles[2],
                        color = "gray", alpha = 0.7, label = "90% Confidence Interval")
                        
        
        plt.axhline(y = self.initial_investment, color = "black", linestyle = "--", 
                    label = "Initial Investment")
        plt.title("Portfolio Monte Carlo Simulation")
        plt.xlabel("Trading Days")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True)
        plt.show()

def run_portfolio_analysis(tickers, weights = None, initial_investment = 100000,
                           num_simulations = 1000, time_horizon = 252,
                           transaction_costs = None, optimization_goal = "sharpe",
                           min_weight = 0, max_weight = 1):
    # Get historical data
    analyzer = PortfolioAnalyzer(tickers, None, initial_investment, transaction_costs)
    stock_data = analyzer.get_stock_data()
    returns = stock_data.pct_change().dropna()
    
    # Optimize portfolio if weights not provided
    if weights is None:
        optimizer = PortfolioOptimizer(returns, analyzer.transaction_costs)
        weights = optimizer.optimize(objective = optimization_goal,
                                     min_weight = min_weight,
                                     max_weight = max_weight)
        print("\nOptimized Portfolio Weights:")
        for ticker, weight in zip(tickers, weights):
            print(f"{ticker}: {weight:.4f}")
    
    analyzer.weights = weights
    
    # Analyze individual stocks
    stock_analyzer = StockAnalyzer(returns, stock_data)
    stock_metrics = stock_analyzer.calculate_metrics()
    print("========== Individual Stock Analysis ==========")
    print(stock_metrics)
    stock_analyzer.plot_metrics(stock_metrics)
    
    # Run Monte Carlo simulation
    simulation_results, returns, distribution_finders = analyzer.run_monte_carlo(
        num_simulations, time_horizon
    )
    
    # Analyze results
    analysis = analyzer.analyze_results(simulation_results, returns, distribution_finders)
    
    # Generate efficient frontier
    optimizer = PortfolioOptimizer(returns, analyzer.transaction_costs)
    efficient_frontier = optimizer.generate_efficient_frontier()
    
    # Plot efficient frontier
    plt.figure(figsize = (10, 6))
    plt.scatter(efficient_frontier['volatility'], efficient_frontier['return'],
               c = efficient_frontier['sharpe'], cmap = 'viridis')
    plt.colorbar(label = 'Sharpe Ratio')
    current_ret, current_vol, current_sharpe = optimizer.calculate_portfolio_metrics(weights)
    plt.scatter(current_vol, current_ret, color = 'red', marker = '*', s = 200,
               label = 'Current Portfolio')
    plt.xlabel('Portfolio Volatility')
    plt.ylabel('Portfolio Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.show()
    
    # Plot Monte Carlo results
    analyzer.plot_results(simulation_results)
    
    return analysis, stock_metrics, efficient_frontier