import pandas as pd, datetime as dt
import yfinance as yf

def load_history(SYMBOL, period=None):

    # print('entering stock data loader')
    # if singular string
    # if isinstance(tickers, str):
    #     tickers = [tickers]

    # Iterate through Tickers
    # data = {}
    # for t in tickers:
    print(f'-- extracting data for: {SYMBOL}')
    # download data, extract approapriate columns
    data = yf.download(SYMBOL, period=period, auto_adjust=False, progress=False)
    data = data[['Open','High','Low','Close','Volume']].dropna()
    data = data.rename(columns={
        'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'
    })
    # Collapse the MultiIndex columns
    data = data.stack(level=1, future_stack=True).reset_index()

    # Rename 'level_1' to 'Ticker'
    data = data.rename(columns={'level_1': 'ticker'})

    # Set Date back as index
    data = data.set_index('Date')
    # df_list.append(data)

    # Combine all dataframes into a single dataframe
    # df = pd.concat(data)
    df = data.copy()
    return df

### MISC
# from datetime import datetime, timedelta
# import pandas as pd
# import yfinance as yf

# # Simple pull; keep Lambda out of a VPC for outbound internet unless you configure NAT

# def load_history(ticker: str, days: int = 60) -> pd.DataFrame:
#     end = datetime.utcnow().date() + timedelta(days=1)
#     start = end - timedelta(days=days)
#     df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
#     df.index = df.index.tz_localize(None)
#     df = df.rename(columns={
#         'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'
#     })
#     return df[['open','high','low','close','volume']]