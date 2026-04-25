"""
Configuration for P2-ETF-HYPERGRAPH-TRANSFORMER engine.
"""

import os
from datetime import datetime

HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-hypergraph-transformer-results"

FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# Static sector grouping for hyperedges
SECTOR_MAP = {
    "Tech": ["XLK", "QQQ", "IWF", "XSD"],
    "Healthcare": ["XLV", "XBI"],
    "Financials": ["XLF"],
    "Energy": ["XLE"],
    "Consumer Discretionary": ["XLY"],
    "Consumer Staples": ["XLP"],
    "Utilities": ["XLU"],
    "Industrials": ["XLI"],
    "Materials": ["XME", "GDX"],
    "Bonds": ["TLT", "LQD", "HYG", "VCIT"],
    "Precious Metals": ["GLD", "SLV"],
    "Real Estate": ["VNQ"]
}

# Hypergraph construction
ROLLING_CORR_WINDOW = 63
MACRO_CORR_THRESHOLD = 0.4
NODE_FEATURE_WINDOW = 5

# Model
HIDDEN_CHANNELS = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1
EPOCHS = 80
BATCH_SIZE = 1              # one day at a time
LEARNING_RATE = 0.001
RANDOM_SEED = 42
MIN_OBSERVATIONS = 252
TRAIN_START = "2008-01-01"

TODAY = datetime.now().strftime("%Y-%m-%d")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
