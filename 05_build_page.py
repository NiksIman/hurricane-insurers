"""Build the public results page: inject out/*.csv results as JSON into page/template.html.

Writes index.html at the repo root (GitHub Pages), plus a copy to argv[1] if given.
"""
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
PAGE = ROOT / "page"

GROUPS = {"FL": ["UVE", "HCI", "HRTG", "ACIC"], "National": ["ALL", "TRV", "CB", "PGR"],
          "Reinsurer": ["RNR", "EG", "ACGL"]}

px = pd.read_csv(ROOT / "data" / "closes.csv", index_col=0, parse_dates=True)
px = px[px["SPY"].notna()].sort_index()
ret = px.pct_change(fill_method=None)
dates = px.index
ev = pd.read_csv(OUT / "events.csv")
ev["T0"] = pd.to_datetime(ev["T0"])
tt = pd.read_csv(OUT / "per_event_ticker.csv")


def r1(x, k=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


# --- cumulative AR paths (vs SPY), rebuilt with the same market model as 03_event_study.py
def cum_path(tk, i0):
    a, b = i0 - 250, i0 - 30
    y, x = ret[tk].iloc[a:b + 1], ret["SPY"].iloc[a:b + 1]
    m = y.notna() & x.notna()
    if m.sum() < 150 or (y[m] == 0).mean() > 0.30 or i0 + 60 >= len(dates):
        return None
    beta, alpha = np.polyfit(x[m], y[m], 1)
    idx = np.arange(i0 - 5, i0 + 61)
    ar = ret[tk].iloc[idx].values - (alpha + beta * ret["SPY"].iloc[idx].values)
    if np.isnan(ar).any():
        return None
    return np.concatenate([[0.0], np.cumsum(ar)]) * 100     # day -6 .. +60


ev["i0"] = [int(np.searchsorted(dates, d)) for d in ev["T0"]]
paths = {(r.storm, tk): cum_path(tk, r.i0) for r in ev.itertuples() for g in GROUPS.values() for tk in g}


def group_mean(storms, grp):
    per_ev = []
    for s in storms:
        ps = [paths[(s, t)] for t in GROUPS[grp] if paths[(s, t)] is not None]
        if ps:
            per_ev.append(np.mean(ps, axis=0))
    return ([r1(v, 2) for v in np.mean(per_ev, axis=0)], len(per_ev)) if per_ev else (None, 0)


panels = []
for key, title, mask in [
    ("fl", "2020–24 · Landfall in Florida", (ev.period == "2020-2024") & ev.fl_landfall),
    ("other", "2020–24 · Landfall anderswo", (ev.period == "2020-2024") & ~ev.fl_landfall),
    ("pre", "Vor 2020 · Vergleich", ev.period == "pre-2020"),
]:
    storms = ev[mask].storm.tolist()
    series = {}
    for g in GROUPS:
        y, n = group_mean(storms, g)
        series[g] = dict(y=y, n=n)
    panels.append(dict(key=key, title=title, storms=storms, series=series))

fl_events = []
for r in ev[ev.fl_landfall].itertuples():
    tick = {t: [r1(v, 2) for v in paths[(r.storm, t)]] for t in GROUPS["FL"] if paths[(r.storm, t)] is not None}
    mean = np.mean([paths[(r.storm, t)] for t in tick], axis=0)
    fl_events.append(dict(storm=r.storm, year=int(r.year), tickers=tick, mean=[r1(v, 2) for v in mean]))

# --- landfall wind from HURDAT2
hd = pd.read_csv(ROOT / "data" / "hurdat2_landfalls.csv", dtype={"date": str, "time_utc": str})
hd["key"] = hd["date"] + hd["time_utc"].str.zfill(4)


def cat(kt, status):
    if status != "HU":
        return {"EX": "post-trop.", "TS": "TS"}.get(status, status)
    return "Kat. " + str(1 if kt < 83 else 2 if kt < 96 else 3 if kt < 113 else 4 if kt < 137 else 5)


events = []
for r in ev.itertuples():
    lf_utc = pd.Timestamp(r.landfall_et).tz_localize("America/New_York").tz_convert("UTC")
    k = lf_utc.strftime("%Y%m%d%H%M")
    h = hd[(hd.storm == r.storm.upper()) & (hd.key == k)]
    kt = int(h.wind_kt.iloc[0]) if len(h) else None
    events.append(dict(storm=r.storm, year=int(r.year), state=r.state, fl=bool(r.fl_landfall),
                       landfall_et=r.landfall_et, T0=r.T0.strftime("%Y-%m-%d"), T0_brief=r.T0_brief,
                       kt=kt, cat=cat(kt, h.status.iloc[0]) if len(h) else "", cluster=int(r.cluster),
                       period=r.period, note=r.note if isinstance(r.note, str) else ""))

# --- per-event FL-basket table (group means)
gm = pd.read_csv(OUT / "per_event_group.csv")
flg = gm[gm.group == "FL"].set_index("storm")
basket_rows = []
for e in events:
    if e["storm"] not in flg.index:
        continue
    g = flg.loc[e["storm"]]
    basket_rows.append(dict(storm=e["storm"], year=e["year"], fl=e["fl"], period=e["period"],
                            pre=r1(g["CARpre[-5,-1]_SPY"] * 100), d1=r1(g["CAR[0,+1]_SPY"] * 100),
                            d5=r1(g["CAR[0,+5]_SPY"] * 100), d20=r1(g["CAR[0,+20]_SPY"] * 100),
                            d60=r1(g["CAR[0,+60]_SPY"] * 100)))

# --- strategy
st = pd.read_csv(OUT / "strategy_trades.csv")
base = st[st.run == "base (T0 = first close after landfall)"]
strat = []
for (s, y), d in base.groupby(["storm", "year"], sort=False):
    d5, d20 = d[d.hold == 5].iloc[0], d[d.hold == 20].iloc[0]
    strat.append(dict(storm=s, year=int(y), fl=bool(d5.fl_landfall), period=d5.period, n=int(d5.n_members),
                      entry=d5.entry, ex5=r1(d5.ex_SPY * 100), ex20=r1(d20.ex_SPY * 100),
                      ex5k=r1(d5.ex_KIE * 100), ex20k=r1(d20.ex_KIE * 100)))

ss = pd.read_csv(OUT / "strategy_summary.csv")
ss = ss[(ss.measure == "ex_SPY")]
strat_sum = [dict(run=r.run, subset=r.subset, hold=int(r.hold), n=int(r.n), mean=r1(r["mean"] * 100),
                  median=r1(r["median"] * 100), hit=r1(r.hit_pos * 100, 0), worst=r1(r.worst * 100),
                  lo=r1(r.ci_lo * 100), hi=r1(r.ci_hi * 100))
             for _, r in ss.iterrows() if r.n > 0]

# --- group summary (selected) and vol
gs = pd.read_csv(OUT / "group_summary.csv")
grp_sum = [dict(subset=r.subset, group=r.group, measure=r.measure, n=int(r.n), clusters=int(r.clusters),
                mean=r1(r["mean"] * 100), lo=r1(r.ci_lo * 100), hi=r1(r.ci_hi * 100), t=r1(r.t, 2),
                hit=r1(r.hit_pos * 100, 0))
           for _, r in gs.iterrows() if r.n > 0 and ("_SPY" in r.measure or r.measure.startswith("dlog"))]
vr = pd.read_csv(OUT / "vol_FL_minus_national.csv")
vol_rel = [dict(subset=r.subset, measure=r.measure, n=int(r.n), mean=r1(r["mean"] * 100),
                lo=r1(r.ci_lo * 100), hi=r1(r.ci_hi * 100)) for _, r in vr.iterrows()]
slope = pd.read_csv(OUT / "overshoot_reversal_slope.csv")
slope = [dict(y=r.y, slope=r1(r.slope, 2), lo=r1(r.ci_lo, 2), hi=r1(r.ci_hi, 2), n=int(r.n)) for _, r in slope.iterrows()]

# --- Milton check
mil = pd.read_csv(OUT / "milton_check.csv", index_col=0)
milton = {str(i)[:10] if i[0].isdigit() else i: {c: r1(mil.loc[i, c] * 100) for c in mil.columns} for i in mil.index}

data = dict(panels=panels, fl_events=fl_events, events=events, basket=basket_rows, strat=strat,
            strat_sum=strat_sum, grp_sum=grp_sum, vol_rel=vol_rel, slope=slope, milton=milton)

tpl = (PAGE / "template.html").read_text(encoding="utf-8")
body = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
head, rest = body.split("</style>", 1)
html = ('<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + head + "[hidden] { display: none !important; }\n</style>\n</head>\n<body>\n"
        + rest.strip() + "\n</body>\n</html>\n")
out = ROOT / "index.html"            # served by GitHub Pages from the repo root
out.write_text(html, encoding="utf-8")
print("wrote", out, f"{len(html)/1024:.0f} KB")
if len(sys.argv) > 1:
    shutil.copy(out, sys.argv[1]); print("copied to", sys.argv[1])
