# P2-ETF-HYPERGRAPH-TRANSFORMER

**Hypergraph Transformer – Sector & Macro Hyperedge Attention for ETF Prediction**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-HYPERGRAPH-TRANSFORMER/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-HYPERGRAPH-TRANSFORMER/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--hypergraph--transformer--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-hypergraph-transformer-results)

## Overview

`P2-ETF-HYPERGRAPH-TRANSFORMER` models higher‑order relationships among ETFs through hyperedges. Static sector groupings and dynamic macro‑correlation groups form a hypergraph, and a multi‑head attention mechanism over these hyperedges captures complex dependencies beyond pairwise graphs. The model predicts next‑day ETF returns and ranks the top picks per universe.

## Methodology

- **Hyperedges**: static sector assignments (e.g., Tech, Healthcare) + dynamic macro‑correlation groups (ETFs with |correlation| > 0.4 with VIX, DXY, etc.)
- **Hypergraph Transformer**: multi‑head attention over hyperedges to update ETF embeddings.
- **Training**: daily snapshots from 2008–2026, one epoch over all days per iteration.
- **Inference**: latest snapshot → next‑day return predictions.

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
text
