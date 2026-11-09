# Hurricanes & Insurers

Out-of-sample test (2020–2024) of “Pricing Poseidon” (Kruttli, Roth Tran, Watugala, *Journal of Finance* 80(2), 2025)
with a focus on Florida-concentrated P&C insurers. Idea: Niki.

**Results page:** https://niksiman.github.io/hurricane-insurers/

Tier 1 only (daily prices). The volatility risk premium (Tier 2, needs historical implied volatility) is **not** tested.

## Summary

| Question | Tier 1 result |
|---|---|
| Do FL insurers underperform before landfall? | **Yes, robust.** CAR[−5,−1] vs SPY averages **−8.4%** over the 6 Florida landfalls (6 of 6 negative). Even for landfalls elsewhere **−3.7%**. National insurers and reinsurers ≈ −1%. |
| Do they reliably recover after landfall? | **No.** CAR[0,+5] +1.1%, [0,+20] +2.6%, far from significant. Strong rebounds after Irma 2017 and Milton 2024, a further **−21%** after Ian 2022. |
| Tradable? (buy FL basket at close T0−1) | **No.** 5 days: mean +2.2% vs SPY, median +0.4%, 3 of 6 positive. 20 days: worst case **−27.9% (Ian)**. With a FedNat-style failure in the basket the 5-day mean turns negative (−1.4%). |
| Uncertainty elevated for 2–3 months? | **Yes, consistent with the paper.** Realized vol of FL insurers vs national insurers +40 log points over days 0–20 (≈ +49%), still +26 over days 21–60. |

Whether prices recover depends on whether the realized damage is smaller than feared. The size of the pre-landfall drop alone does not predict the rebound
(slope −0.32, 95% CI [−0.61, +0.17]).

## Method

- **Prices:** Yahoo Finance adjusted daily closes, 2004 – June 2025 (not committed; `01_download.py` fetches them). ACIC carries the UIHC history, EG the RE history. FedNat (FNHC) is delisted and unavailable, so it enters only as a −100% / −50% scenario for Ian.
- **Landfalls:** NHC HURDAT2 best track (`02_landfalls.py`). **T0 = first NYSE close after landfall** (UTC → New York). Differs from the brief's dates for six storms that made landfall after the close or on a weekend; the brief's convention is included as a robustness run.
- **Abnormal returns:** market model on days [−250, −30], vs SPY and vs KIE. Stocks with > 30% zero-return days in the estimation window are excluded; this only affects the pre-2020 comparison (UVE for Katrina 2005, ACIC/UIHC for Ike, Irene and Sandy).
- **Overlap:** events with T0 ≤ 20 trading days apart form a cluster. 16 events → 7 clusters. Cluster-robust SEs and cluster-bootstrap 95% intervals.

## Limits

Six Florida landfalls in five clusters → case evidence, not statistics. Confounds: Florida reform 2022/23, hard reinsurance market, earnings (HRTG around Debby), market shocks (Debby = 5 Aug 2024; Ike = Lehman day). Survivorship: UPC sits inside ACIC, FedNat is missing.

## Reproduce

```bash
pip install pandas numpy yfinance matplotlib
python 01_download.py      # prices -> data/
python 02_landfalls.py     # HURDAT2 landfall records -> data/
python 03_event_study.py   # event study, strategy, charts -> out/
python 05_build_page.py    # results page -> index.html
python 04_print_tables.py summary   # or: strategy | trades | fltickers | vol | slope | milton
```

| Path | Content |
|---|---|
| `out/events.csv` | events, landfall in ET, T0, brief's T0, cluster |
| `out/per_event_ticker.csv` | every stock × event: CARs (SPY & KIE), reversal, volatility |
| `out/per_event_group.csv`, `out/group_summary.csv` | group means per event; averages with cluster SEs and bootstrap CIs |
| `out/strategy_trades.csv`, `out/strategy_summary.csv` | basket strategy, all runs |
| `out/*.png` | static charts |
| `page/template.html` | page template; `index.html` is generated from it |

Not investment advice.
