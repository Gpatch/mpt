import pandas as pd
import numpy as np
import math
import itertools
import random

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

def generate_portfolios(portfolio, risk_free_rate):
    returns = []
    risks = []
    sharpe_ratios = []

    weights_combinations = np.array([comb for comb in itertools.product(np.arange(0, 1 + 0.05, 0.05), repeat=len(portfolio))
                        if np.isclose(sum(comb), 1.0)])
    if len(weights_combinations) > 20000:
        weights_combinations = weights_combinations[:20000]
        random.shuffle(weights_combinations)

    for weights in weights_combinations:
        p_return = portfolio_e_return(portfolio, weights)
        p_risk = portfolio_risk(portfolio, weights)
        p_ratio = sharpe_ratio(p_return, risk_free_rate, p_risk)

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




def search_stocks(stocks_df, desired_sharpe_ratio, risk_free_rate, timeframe, start_date, end_date):
    all_stocks = stocks_df['Brand_Name'].unique()
    stock_combinations = list(itertools.combinations(all_stocks, 5))
    random.shuffle(stock_combinations)
    counter = 0
    for combination in stock_combinations:
        counter +=1
        print(f"Processing portfolio number {counter}: {combination}")
        data_retrieval.clean_and_save_stocks(combination, timeframe, start_date, end_date)
        df = pd.read_csv('selected_stocks_data.csv')
        portfolio = portfolio_as_array(df)
        gen_portfolios = generate_portfolios(portfolio, risk_free_rate)
        combination_max_sharpe = np.max(gen_portfolios[2])
        combination_max_return = np.max(gen_portfolios[0])
        if combination_max_sharpe >= desired_sharpe_ratio:
            print("Stocks: " + str(combination))
            print("Sharpe ratio: " + str(combination_max_sharpe))
            print("Returns: "+ str(combination_max_return))
            print("Weights: " + str(gen_portfolios[3]))
            return gen_portfolios
        else:
            print(f"Unsuitable portfolio, current maximum sharpe ratio {combination_max_sharpe} < {desired_sharpe_ratio}")
            print("--------------------------------------------------")
    return []

def search_portfolios(stocks, timeframe, start_date, end_date):
    data_retrieval.clean_and_save_stocks(stocks, timeframe, start_date, end_date)
    df = pd.read_csv('selected_stocks_data.csv')
    portfolio = portfolio_as_array(df)
    gen_portfolios = generate_portfolios(portfolio, timeframe)

    return gen_portfolios