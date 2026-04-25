"""
Data loading and hypergraph building for Hypergraph Transformer.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
import torch
import config

def load_master_data():
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    path = hf_hub_download(
        repo_id=config.HF_DATA_REPO, filename=config.HF_DATA_FILE,
        repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
    )
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_returns_matrix(df_wide, tickers):
    available = [t for t in tickers if t in df_wide.columns]
    df_long = df_wide.melt(id_vars=['Date'], value_vars=available,
                           var_name='ticker', value_name='price')
    df_long = df_long.sort_values(['ticker', 'Date'])
    df_long['log_return'] = df_long.groupby('ticker')['price'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df_long = df_long.dropna(subset=['log_return'])
    return df_long.pivot(index='Date', columns='ticker', values='log_return')[available].dropna()

def prepare_macro(df_wide):
    macro_cols = [c for c in config.MACRO_COLS if c in df_wide.columns]
    macro_df = df_wide[['Date'] + macro_cols].copy()
    macro_df = macro_df.set_index('Date').ffill().dropna()
    return macro_df

def build_hypergraph_sequence(returns, macro):
    """
    Build a list of daily hypergraph snapshots.
    Each snapshot is a dict:
        x: tensor (num_etfs, feat_dim) – ETF node features
        hyperedge_index: list of lists, each sublist contains indices of ETFs belonging to that hyperedge
        y: tensor (num_etfs,) – next-day returns
    """
    common = returns.index.intersection(macro.index)
    returns = returns.loc[common]
    macro = macro.loc[common]
    tickers = returns.columns.tolist()
    n_etfs = len(tickers)
    n_macro = len(macro.columns)

    ret_scaler = StandardScaler().fit(returns.values.reshape(-1, 1))
    macro_scaler = StandardScaler().fit(macro.values)

    # Precompute static sector hyperedges (indices)
    sector_hedges = []
    for sector, tkr_list in config.SECTOR_MAP.items():
        indices = [tickers.index(t) for t in tkr_list if t in tickers]
        if len(indices) >= 2:   # hyperedge needs at least 2 members
            sector_hedges.append(indices)

    snapshots = []
    for idx in range(len(returns) - 1):
        # ETF node features: past 5-day returns + current macro values
        node_feats = []
        for tkr in tickers:
            ret_series = returns[tkr]
            if idx >= config.NODE_FEATURE_WINDOW - 1:
                window = ret_series.iloc[idx - config.NODE_FEATURE_WINDOW + 1: idx + 1].values
            else:
                window = ret_series.iloc[:idx + 1].values
                if len(window) < config.NODE_FEATURE_WINDOW:
                    window = np.pad(window, (config.NODE_FEATURE_WINDOW - len(window), 0), 'edge')
            window_scaled = ret_scaler.transform(window.reshape(-1, 1)).flatten()
            macro_vals = macro_scaler.transform(macro.iloc[idx].values.reshape(1, -1)).flatten()
            feat = np.concatenate([window_scaled, macro_vals])
            node_feats.append(feat)

        x = torch.tensor(np.stack(node_feats), dtype=torch.float32)

        # Dynamic macro hyperedges
        macro_hedges = []
        if idx >= config.ROLLING_CORR_WINDOW - 1:
            rolling_ret = returns.iloc[idx - config.ROLLING_CORR_WINDOW + 1: idx + 1]
            rolling_macro = macro.iloc[idx - config.ROLLING_CORR_WINDOW + 1: idx + 1]
            for col in macro.columns:
                corr = rolling_ret.apply(lambda etf_ret: etf_ret.corr(rolling_macro[col]))
                # select ETFs with absolute correlation > threshold
                selected = [i for i, tkr in enumerate(tickers) if abs(corr[tkr]) > config.MACRO_CORR_THRESHOLD]
                if len(selected) >= 2:
                    macro_hedges.append(selected)

        # Combine static and dynamic hyperedges
        hyperedge_index = sector_hedges + macro_hedges

        # Target
        y = torch.tensor(returns.iloc[idx + 1].values, dtype=torch.float32)

        snapshots.append({
            'x': x,
            'hyperedge_index': hyperedge_index,
            'y': y
        })
    return snapshots
