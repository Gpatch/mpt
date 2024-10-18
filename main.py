import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools

import data_retrieval

def asset_pct_returns(df, ticker):
    asset = df[df['Ticker'] == ticker]['Close']
    returns = asset.pct_change().to_numpy()[1:]
    return returns

def asset_e_return(df, ticker):
    asset_return = asset_pct_returns(df, ticker).mean()
    return asset_return

def portfolio_as_array(df):
    portfolio = {}
    unique_tickers = df['Ticker'].unique()
    for t in unique_tickers:
        asset_returns = asset_pct_returns(df, t)
        portfolio[t] = asset_returns
    return portfolio

def portfolio_avgs_as_array(portfolio):
    avgs = []
    for asset in portfolio:
        avg = portfolio[asset].mean()
        avgs.append(avg)
    return avgs
        


def portfolio_e_return(portfolio, weights):
    total_sum = 0
    avg_returns = portfolio_avgs_as_array(portfolio)
    for asset_return, weight in zip(avg_returns, weights):
        total_sum += (asset_return * weight)
    return total_sum


def portfolio_risk(portfolio, weights):
    min_length = min(len(value) for value in portfolio.values())

    # Truncate all arrays to the minimum length
    for key, value in portfolio.items():
        portfolio[key] = value[:min_length]

    portfolio_df = pd.DataFrame(portfolio, columns=portfolio.keys())
    cov_matrix = portfolio_df.cov()
    variance = weights.T @ cov_matrix @ weights
    std = math.sqrt(variance)
    return std

def sharpe_ratio(p_return, r_free, risk):
    return (p_return - r_free) / risk

def generate_portfolios(portfolio):
    returns = []
    risks = []
    sharpe_ratios = []

    weights_combinations = np.array([comb for comb in itertools.product(np.arange(0, 1 + 0.05, 0.05), repeat=len(portfolio))
                        if np.isclose(sum(comb), 1.0)])

    for weights in weights_combinations:
        p_return = portfolio_e_return(portfolio, weights)
        p_risk = portfolio_risk(portfolio, weights)
        p_ratio = sharpe_ratio(p_return, 0.003077, p_risk)

        returns.append(p_return)
        risks.append(p_risk)
        sharpe_ratios.append(p_ratio)

    max_sharpe_index = np.argmax(sharpe_ratios)
    max_weights = weights_combinations[max_sharpe_index]
    
    return returns, risks, sharpe_ratios, max_weights

def efficient_frontier(returns, risks):
    df = pd.DataFrame({'Returns': returns, 'Risk': risks})
    df['Risk'] = df['Risk'].round(2)

    frontier_returns = []
    frontier_risks = []
    unique_risks = np.unique(np.round(risks, decimals=2))

    for risk in unique_risks:
        max_return = df.loc[df['Risk'] == risk, 'Returns'].max()
        frontier_returns.append(max_return)
        frontier_risks.append(risk)

    return np.array(frontier_returns), np.array(frontier_risks)


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


def main():
    stocks_df = pd.read_csv('stocks_data.csv')
    no_peloton_df = stocks_df.loc[stocks_df['Brand_Name'] != 'peloton']
    all_stocks = no_peloton_df['Brand_Name'].unique()
    stock_combinations = list(itertools.combinations(all_stocks, 5))
    flag = False
    counter = 0
    for combination in stock_combinations:
        counter +=1
        print(counter)
        
        data_retrieval.clean_and_save_stocks(combination)
        df = pd.read_csv('selected_stocks_data.csv')
        portfolio = portfolio_as_array(df)
        gen_portfolios = generate_portfolios(portfolio)
        combination_max_sharpe = np.max(gen_portfolios[2])
        combination_max_return = np.max(gen_portfolios[0])
        if combination_max_sharpe >= 1.5:
            flag = True
            print("stocks: " + str(combination))
            print("Sharpe ratio: " + str(combination_max_sharpe))
            print("return: "+ str(combination_max_return))
            print("weights: " + str(gen_portfolios[3]))
            frontier_returns, frontier_risks = efficient_frontier(gen_portfolios[0], gen_portfolios[1])
            plot_portfolios(gen_portfolios[0], gen_portfolios[1], gen_portfolios[2], frontier_returns, frontier_risks)
        if flag:
            break

main()

