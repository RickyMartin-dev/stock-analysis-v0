import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

WINDOW=14

def _rsi(series: pd.Series, n=10):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = -delta.clip(upper=0).rolling(n).mean()
    rs = (up / (down + 1e-9)).fillna(0)
    return 100 - (100 / (1 + rs))

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['ret'] = d['close']/d['open'] - 1.0
    d['ret_3'] = d['close'].pct_change(3)
    d['ret_5'] = d['close'].pct_change(5)
    d['ma_5'] = d['close'].rolling(5).mean()
    d['ma_10'] = d['close'].rolling(10).mean()
    d["std_5"] = d["close"].rolling(5).std()
    d['vol10'] = d['ret'].rolling(10).std()
    d["hl_range"] = (d["high"] - d["low"]) / d["close"].shift(1)
    d["v_ma_5"] = d["volume"].rolling(5).mean()
    d["v_std_5"] = d["volume"].rolling(5).std()
    d['amp'] = (d['high']-d['low'])/d['open']
    d['ret_roll5_mean'] = d['ret'].rolling(5).mean()
    d['ret_roll5_std'] = d['ret'].rolling(5).std().fillna(0)
    d['rsi10'] = _rsi(d['close'],10)

    d["dow"] = d.index.dayofweek
    dow = pd.get_dummies(d["dow"], prefix="dow")
    dow.columns = [f"dow_{int(col.split('_')[1])}" for col in dow.columns]
    
    # concat
    d = pd.concat([d, dow], axis=1)

    FEATURE_COLS = [
            'ret','ret_3','ret_5','ma_5','ma_10','std_5',
            'vol10','hl_range','v_ma_5','v_std_5','amp','ret_roll5_mean',
            'ret_roll5_std','open','high','low','close','volume',
        ] + [c for c in d.columns if c.startswith("dow_")]

    iterate_cols = FEATURE_COLS.copy()
    # lag features
    feat_cols = []
    for c in iterate_cols:
        # print(c)
        for lag in range(1, WINDOW+1):
            name = f"{c}_lag{lag}"
            d[name] = d[c].shift(lag)
            FEATURE_COLS.append(name)

    df = d.copy()
    return df.dropna().copy(), FEATURE_COLS

def make_label(df: pd.DataFrame) -> pd.Series:
    return (df['close'] > df['open']).astype(int)

def windowize(FEATURE_COLS, df_feat: pd.DataFrame, scaler: StandardScaler|None=None, fit=False):
    X_list, y_list = [], []

    y = make_label(df_feat)
    for i in range(WINDOW, len(df_feat)):
        chunk = df_feat.iloc[i-WINDOW:i]
        x = chunk[FEATURE_COLS].values
        x = x.reshape(-1)  # flatten window
        # dow = chunk.index[-1].weekday()
        # dow_onehot = np.zeros(5); dow_onehot[dow if dow<5 else 4] = 1
        # x = np.concatenate([x, dow_onehot])
        X_list.append(x); y_list.append(int(y.iloc[i]))

    X = np.array(X_list); y = np.array(y_list)
    # print(X)
    if fit:
        scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X) if scaler else X

    return Xs, y, scaler

def prepare_training(df: pd.DataFrame):
    df_feat, cols = add_features(df)
    X, y, scaler = windowize(cols, df_feat, fit=True)
    return X, y, scaler

def prepare_inference(df: pd.DataFrame, scaler: StandardScaler):
    df_feat, cols = add_features(df)
    X, _, _ = windowize(cols, df_feat, scaler=scaler, fit=False)
    if len(X)==0:
        raise ValueError("Not enough data to build a 14‑day window")
    return X[-1]
    
### MISC
# import numpy as np
# import pandas as pd

# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
# import joblib

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset



# # build_features: for single to multiple tickers to X,Y for training
# # expected df columns per ticker: ['open','high','low','close','volume'] with DatetimeIndex

# def build_features(d, lookback: int = 14):

#     lookback = int(lookback)
#     print(f'creating features, with lookback: {type(lookback)}{lookback}')
#     # rows, labels = [], []

#     # iterate through data
#     # for tkr, d in df.items():
#     # print(f'  creating features for: {tkr}')

#     # feature creation
#     d = d.copy().sort_index()

#     d['ret_1'] = d['Close'].pct_change()
#     d['ret_3'] = d['Close'].pct_change(3)
#     d['ret_5'] = d['Close'].pct_change(5)
#     d['ma_5'] = d['Close'].rolling(5).mean()
#     d['ma_10'] = d['Close'].rolling(10).mean()
#     d["std_5"] = d["Close"].rolling(5).std()
#     d['vol10'] = d['ret_1'].rolling(10).std()
#     d["hl_range"] = (d["High"] - d["Low"]) / d["Close"].shift(1)
#     d["v_ma_5"] = d["Volume"].rolling(5).mean()
#     d["v_std_5"] = d["Volume"].rolling(5).std()
#     d['amp'] = (d['High']-d['Low'])/d['Open']
#     d['rsi10'] = _rsi(d['Close'],10)

#     # day of the week encoding with flat column names
#     d["dow"] = d.index.dayofweek
#     dow = pd.get_dummies(d["dow"], prefix="dow")
#     dow.columns = [f"dow_{int(col.split('_')[1])}" for col in dow.columns]
    
#     # concat
#     d = pd.concat([d, dow], axis=1)
#     # label: close > open (same day)
#     d["label"] = (d["Close"] > d["Open"]).astype(int)

#     base_feats = [
#             'ret_1','ret_3','ret_5','ma_5','ma_10','std_5',
#             'vol10','hl_range','v_ma_5','v_std_5','amp','rsi10',
#             'Volume',
#         ] + [c for c in d.columns if c.startswith("dow_")]

#     print('    creating lag features')
#     # lag features
#     feat_cols = []
#     for c in base_feats:
#         for lag in range(1, lookback+1):
#             name = f"{c}_lag{lag}"
#             d[name] = d[c].shift(lag)
#             feat_cols.append(name)

#     # create/define x, y
#     d = d.dropna().copy()
#     x = d[feat_cols].astype(float)
#     y = d["label"].astype(int)
    
#     # print('    add to X, Y')
#     # # add to appropriate list
#     # rows.append(x)
#     # labels.append(y)

#     # print(f'  finished feature creation')
#     # X = pd.concat(rows).dropna().values
#     # Y = pd.concat(labels).loc[pd.Index(pd.concat(rows).dropna().index)].values
#     # return X.astype('float32'), Y.astype('int64')
#     return x, y

# def create_data_set_part1(X, y, val_friction: float = 0.2):

#     # parameters
#     n = len(X)

#     # Split info
#     split = int(n * (1 - val_friction))
#     X_tr, X_val = X.iloc[:split], X.iloc[split:]
#     y_tr, y_val = y.iloc[:split], y.iloc[split:]

#     # Preprocess (StandardScaler)
#     numeric_features = X.columns.tolist()
#     pre = ColumnTransformer([("num", StandardScaler(), numeric_features)], remainder="drop")
#     pre_pipe = Pipeline([("pre", pre)])    

#     X_train_set = pre_pipe.fit_transform(X_tr)
#     X_val_set = pre_pipe.transform(X_val)

#     # X_train_set.shape, X_val_set.shape 

#     return X_train_set, y_tr, X_val_set, y_val, pre_pipe

# def create_data_set_part2(Xtrain, ytrain, Xval, yval, batchsize):
    
#     tr_ds = TensorDataset(torch.from_numpy(Xtrain).float(), torch.from_numpy(ytrain.values).float())
#     va_ds = TensorDataset(torch.from_numpy(Xval).float(), torch.from_numpy(yval.values).float())
#     tr_dl = DataLoader(tr_ds, batch_size=batchsize, shuffle=True)
#     va_dl = DataLoader(va_ds, batch_size=batchsize)

#     return tr_dl, va_dl