"""
Main training script for Hypergraph Transformer engine.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from hypergraph_transformer_model import HGRunner
import push_results

def run_hypergraph():
    print(f"=== P2-ETF-HYPERGRAPH-TRANSFORMER Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master = df_master[df_master['Date'] >= config.TRAIN_START]
    macro = data_manager.prepare_macro(df_master)

    all_results = {}
    top_picks = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        snaps = data_manager.build_hypergraph_sequence(returns, macro)
        if len(snaps) < config.MIN_OBSERVATIONS:
            continue

        in_channels = snaps[0]['x'].size(1)
        runner = HGRunner(in_channels, config.HIDDEN_CHANNELS, config.NUM_HEADS,
                          config.NUM_LAYERS, config.DROPOUT, config.LEARNING_RATE, config.RANDOM_SEED)

        print(f"  Training Hypergraph Transformer on {len(snaps)} days...")
        runner.train_snapshots(snaps, epochs=config.EPOCHS)

        preds = runner.predict_latest(snaps)
        universe_results = {}
        for i, ticker in enumerate(tickers):
            universe_results[ticker] = {"ticker": ticker, "forecast": float(preds[i])}

        all_results[universe_name] = universe_results
        sorted_items = sorted(universe_results.items(), key=lambda x: x[1]["forecast"], reverse=True)
        top_picks[universe_name] = [{"ticker": t, "forecast": d["forecast"]} for t, d in sorted_items[:3]]

    output_payload = {
        "run_date": config.TODAY,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_") and k.isupper() and k != "HF_TOKEN"},
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_hypergraph()
