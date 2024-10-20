import pandas as pd

def import_data(file_path: str):
    df = pd.read_csv(file_path)
    new_df = df.loc[df['Brand_Name'] != 'peloton']
    return new_df

def retrieve__stocks(df, stocks):
    stocks_df = df.loc[df['Brand_Name'].isin(stocks)]
    return stocks_df


def clean_df(df, timeframe, start_date, end_date):
    new_df = df[['Date', 'Close', 'Brand_Name', 'Ticker']]
    no_dupes_df = new_df.drop_duplicates(subset=['Date', 'Ticker']).copy()
    no_dupes_df['Date'] = pd.to_datetime(no_dupes_df['Date'], utc=True)
    filtered_dates = no_dupes_df.loc[(no_dupes_df['Date'] >= start_date) & (no_dupes_df['Date'] <= end_date)]
    if timeframe == 0:
        timeframe_dates = filtered_dates[(filtered_dates['Date'].dt.day == 1)]
    else:
        timeframe_dates = filtered_dates[(filtered_dates['Date'].dt.day == 1) & (filtered_dates['Date'].dt.month == 2)]
    sorted_dates = timeframe_dates.sort_values(by='Date', ascending=True).copy()
    sorted_dates.reset_index(inplace=True, drop=True)
    return sorted_dates


def save_cleaned_df(df):
    df.to_csv("selected_stocks_data.csv", index=False)


def clean_and_save_stocks(stocks, timeframe, start_date, end_date):
    df = import_data("stocks_data.csv")
    stocks_df = retrieve__stocks(df, stocks)
    cleaned_df = clean_df(stocks_df, timeframe, start_date, end_date)
    save_cleaned_df(cleaned_df)