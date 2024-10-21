import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


import data_retrieval
import portfolio_optimizer as mpt

def plot_portfolios(portfolios, risks, sharpe_ratios, frontier_returns, frontier_risks):
    plt.figure(figsize=(10, 6))

    # Create the scatter plot using Seaborn
    scatter = plt.scatter(risks, portfolios, c=sharpe_ratios, cmap='viridis')
    plt.plot(frontier_risks, frontier_returns, color='red', label='Efficient Frontier', linewidth=2)
    plt.colorbar(scatter, label='Sharpe Ratio')

    # Dynamically set the x and y ticks based on data range with 0.1 step size
    plt.xticks(np.arange(np.floor(min(risks)), np.ceil(max(risks)) + 0.1, 0.1))
    plt.yticks(np.arange(np.floor(min(portfolios)), np.ceil(max(portfolios)) + 0.1, 0.1))

    # Find the index of the point with the highest Sharpe ratio
    max_sharpe_idx = np.argmax(sharpe_ratios)
    max_sharpe_risk = risks[max_sharpe_idx]
    max_sharpe_return = portfolios[max_sharpe_idx]
    max_sharpe_ratio = sharpe_ratios[max_sharpe_idx]

    # Highlight the point with the highest Sharpe ratio
    plt.scatter(max_sharpe_risk, max_sharpe_return, color='red', s=100, edgecolor='black', label='Max Sharpe Ratio')

    # Add text annotation for the highest Sharpe ratio point
    plt.text(max_sharpe_risk, max_sharpe_return,
             f'Return: {max_sharpe_return:.2f}, Risk: {max_sharpe_risk:.2f}, Sharpe: {max_sharpe_ratio:.2f}',
             fontsize=9, ha='left', va='bottom', color='black')

    # Labels and title
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.title('Portfolio Risk vs. Return')

    # Show the plot with the highlight
    plt.show()


def main(stocks_choice=None, desired_sharpe_ratio=1.5, timeframe=1, start_date='2020-05-01', end_date='2024-02-01'):
    # Removing outlier stock due to providing non future proof results
    risk_free_rate = 0
    if timeframe == 0:
        risk_free_rate = 0.00333
    elif timeframe == 1:
        risk_free_rate = 0.04
    else:
        raise Exception("Please provide a valid timeframe! (0 or 1 for monthly/annual)")

    stocks_df = data_retrieval.import_data('stocks_data.csv')
    portfolios = []
    

    if stocks_choice is None:
        portfolios = mpt.search_stocks(stocks_df, desired_sharpe_ratio, risk_free_rate, timeframe, start_date, end_date)
    else:
        portfolios = mpt.search_portfolios(stocks_choice, timeframe, start_date, end_date)
        
    frontier_returns, frontier_risks = mpt.efficient_frontier(portfolios[0], portfolios[1])
    plot_portfolios(portfolios[0], portfolios[1], portfolios[2], frontier_returns, frontier_risks)

    

main(start_date='2010-01-01', end_date='2016-01-01')

