import pandas as pd

def import_data(file_path: str):
    df = pd.read_csv(file_path)
    new_df = df.loc[df['Brand_Name'] != 'peloton']
    return new_df

def retrieve__stocks(df, stocks):
    all_stocks = df['Brand_Name'].unique()
    #stock_names = ['nvidia', 'amazon', 'microsoft', 'ubisoft', 'salesforce / slack']
    stocks_df = df.loc[df['Brand_Name'].isin(stocks)]
    return stocks_df


def clean_df(df):
    new_df = df[['Date', 'Close', 'Brand_Name', 'Ticker']]
    no_dupes_df = new_df.drop_duplicates(subset=['Date', 'Ticker']).copy()
    no_dupes_df['Date'] = pd.to_datetime(no_dupes_df['Date'], utc=True)
    filtered_dates = no_dupes_df.loc[no_dupes_df['Date'] >= '2021-01-01']
    monthly_dates = filtered_dates[(filtered_dates['Date'].dt.day == 1)]
    sorted_dates = monthly_dates.sort_values(by='Date', ascending=True).copy()
    sorted_dates.reset_index(inplace=True, drop=True)
    return sorted_dates


def save_cleaned_df(df):
    df.to_csv("selected_stocks_data.csv", index=False)


def clean_and_save_stocks(stocks):
    df = import_data("stocks_data.csv")
    stocks_df = retrieve__stocks(df, stocks)
    cleaned_df = clean_df(stocks_df)
    save_cleaned_df(cleaned_df)