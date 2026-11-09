"""Tier 1, step 1: download daily adjusted closes and save the raw data to CSV.

Research project: hurricane uncertainty and P&C insurer returns (idea: Niki).
Source: Yahoo Finance via yfinance (auto_adjust=True -> split/dividend-adjusted closes).
Range starts 2004 so the optional pre-2020 events (Katrina 2005 ...) get an estimation window.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

TICKERS = [
    # FL-concentrated
    "UVE", "HCI", "HRTG", "ACIC", "UIHC", "FNHC",
    # National P&C
    "ALL", "TRV", "CB", "PGR",
    # Reinsurers
    "RNR", "EG", "RE", "ACGL",
    # Benchmarks
    "SPY", "KIE",
]
START, END = "2004-01-01", "2025-07-01"

closes = {}
for t in TICKERS:
    df = yf.download(t, start=START, end=END, auto_adjust=True, progress=False)
    if df is None or df.empty:
        print(f"{t:5s}: NO DATA")
        continue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv(RAW / f"{t}.csv")
    closes[t] = df["Close"]
    print(f"{t:5s}: {df.index[0].date()} .. {df.index[-1].date()}  n={len(df)}")

panel = pd.DataFrame(closes)
panel.index.name = "Date"
panel.to_csv(ROOT / "data" / "closes.csv")
print("saved", ROOT / "data" / "closes.csv", panel.shape)
