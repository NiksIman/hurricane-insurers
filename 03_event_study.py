"""Tier 1, step 3: event study around US hurricane landfalls, P&C insurer focus.

Inputs : data/closes.csv (01_download.py), landfall times from HURDAT2 (02_landfalls.py).
Outputs: out/*.csv tables, out/*.png charts. Price side only (no implied vol / VRP).

T0 (primary)  = first NYSE close AFTER the landfall time (HURDAT2, UTC -> New York time).
T0 (variant)  = date as written in the brief (next trading day if non-trading).
Relative day k = k-th trading day after T0; [-5,-1] = the five closes before T0.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261109)

GROUPS = {
    "FL": ["UVE", "HCI", "HRTG", "ACIC"],
    "National": ["ALL", "TRV", "CB", "PGR"],
    "Reinsurer": ["RNR", "EG", "ACGL"],
}
TICK2GRP = {t: g for g, ts in GROUPS.items() for t in ts}
ALL_T = [t for ts in GROUPS.values() for t in ts]

# storm, state, primary landfall (UTC, HURDAT2 'L' record), date written in brief, FL landfall, note
EVENTS = [
    ("Katrina",  "LA", "2005-08-29 11:10", "2005-08-29", False, "first FL landfall 25 Aug (Cat 1)"),
    ("Ike",      "TX", "2008-09-13 07:00", "2008-09-13", False, "T0 = Lehman day (15 Sep 2008)"),
    ("Irene",    "NC", "2011-08-27 12:00", "2011-08-27", False, ""),
    ("Sandy",    "NJ", "2012-10-29 23:30", "2012-10-29", False, "NYSE closed 29-30 Oct"),
    ("Harvey",   "TX", "2017-08-26 03:00", "2017-08-25", False, ""),
    ("Irma",     "FL", "2017-09-10 13:00", "2017-09-10", True,  ""),
    ("Michael",  "FL", "2018-10-10 17:30", "2018-10-10", True,  "FL panhandle"),
    ("Hanna",    "TX", "2020-07-25 22:00", "2020-07-25", False, ""),
    ("Isaias",   "NC", "2020-08-04 03:10", "2020-08-04", False, "landfall 23:10 ET 3 Aug"),
    ("Laura",    "LA", "2020-08-27 06:00", "2020-08-27", False, ""),
    ("Sally",    "AL", "2020-09-16 09:45", "2020-09-16", False, ""),
    ("Delta",    "LA", "2020-10-09 23:00", "2020-10-09", False, "landfall Fri 19:00 ET"),
    ("Zeta",     "LA", "2020-10-28 21:00", "2020-10-28", False, "landfall 17:00 ET"),
    ("Ida",      "LA", "2021-08-29 16:55", "2021-08-29", False, ""),
    ("Nicholas", "TX", "2021-09-14 05:30", "2021-09-14", False, ""),
    ("Ian",      "FL", "2022-09-28 19:05", "2022-09-28", True,  "landfall 15:05 ET, before close; FedNat liquidated 27 Sep"),
    ("Nicole",   "FL", "2022-11-10 07:45", "2022-11-10", True,  ""),
    ("Idalia",   "FL", "2023-08-30 11:45", "2023-08-30", True,  "after FL reform re-rating"),
    ("Beryl",    "TX", "2024-07-08 08:40", "2024-07-08", False, ""),
    ("Debby",    "FL", "2024-08-05 11:00", "2024-08-05", True,  "T0 = 5 Aug 2024 global sell-off (yen carry)"),
    ("Francine", "LA", "2024-09-11 22:00", "2024-09-11", False, "landfall 18:00 ET"),
    ("Helene",   "FL", "2024-09-27 03:10", "2024-09-26", True,  "landfall 23:10 ET 26 Sep"),
    ("Milton",   "FL", "2024-10-10 00:30", "2024-10-09", True,  "landfall 20:30 ET 9 Oct"),
]

CAR_WINDOWS = {"pre[-5,-1]": (-5, -1), "[0,+1]": (0, 1), "[0,+5]": (0, 5),
               "[0,+20]": (0, 20), "[0,+60]": (0, 60)}
EST = (-250, -30)
MIN_EST = 150

# ---------------------------------------------------------------- data
px = pd.read_csv(ROOT / "data" / "closes.csv", index_col=0, parse_dates=True)
px = px[px["SPY"].notna()].sort_index()
ret = px.pct_change(fill_method=None)
dates = px.index
close_et = (dates + pd.Timedelta(hours=16)).tz_localize("America/New_York")


def t0_primary(landfall_utc):
    lf = pd.Timestamp(landfall_utc, tz="UTC").tz_convert("America/New_York")
    return int(np.searchsorted(close_et, lf, side="right"))


def t0_brief(date_str):
    return int(np.searchsorted(dates, pd.Timestamp(date_str), side="left"))


def build_events(variant):
    rows = []
    for name, st, lf, bd, fl, note in EVENTS:
        i = t0_primary(lf) if variant == "primary" else t0_brief(bd)
        lf_et = pd.Timestamp(lf, tz="UTC").tz_convert("America/New_York")
        rows.append(dict(storm=name, year=int(lf[:4]), state=st, fl_landfall=fl,
                         landfall_et=lf_et.strftime("%Y-%m-%d %H:%M"),
                         T0=dates[i].date(), i0=i,
                         period="2020-2024" if int(lf[:4]) >= 2020 else "pre-2020", note=note))
    ev = pd.DataFrame(rows)
    # clusters: chain events whose T0s are <= 20 trading days apart (overlapping [0,+20] windows)
    cl, prev = [], None
    for _, r in ev.iterrows():
        cl.append(cl[-1] if prev is not None and r.i0 - prev <= 20 else (cl[-1] + 1 if cl else 1))
        prev = r.i0
    ev["cluster"] = cl
    ev["overlap_20d"] = ev.groupby("cluster")["storm"].transform("count") > 1
    return ev


def market_model(tk, bm, i0):
    a, b = i0 + EST[0], i0 + EST[1]
    if a < 1:
        return None
    y, x = ret[tk].iloc[a:b + 1], ret[bm].iloc[a:b + 1]
    m = y.notna() & x.notna()
    if m.sum() < MIN_EST:
        return None
    if (y[m] == 0).mean() > 0.30:      # illiquid (e.g. UVE 2005 at 1-2 cents): no usable returns
        return None
    beta, alpha = np.polyfit(x[m], y[m], 1)
    return alpha, beta


def ar_path(tk, bm, i0, lo=-60, hi=60):
    mm = market_model(tk, bm, i0)
    if mm is None or i0 + hi >= len(dates):
        return None
    alpha, beta = mm
    idx = np.arange(i0 + lo, i0 + hi + 1)
    ar = ret[tk].iloc[idx].values - (alpha + beta * ret[bm].iloc[idx].values)
    return pd.Series(ar, index=np.arange(lo, hi + 1))


def wsum(s, a, b):
    v = s.loc[a:b]
    return v.sum() if v.notna().all() else np.nan


def ann_vol(v):
    v = v[~np.isnan(v)]
    if len(v) < 15:
        return np.nan
    s = np.std(v, ddof=1) * np.sqrt(252)
    return s if s > 0 else np.nan          # untraded stretches (UVE 2005) -> no vol estimate


def event_ticker_table(ev):
    rows, paths = [], {}
    for _, e in ev.iterrows():
        for tk in ALL_T:
            rec = dict(storm=e.storm, year=e.year, period=e.period, fl_landfall=e.fl_landfall,
                       cluster=e.cluster, ticker=tk, group=TICK2GRP[tk])
            ok = False
            for bm in ("SPY", "KIE"):
                p = ar_path(tk, bm, e.i0)
                if p is None:
                    continue
                ok = True
                for k, (a, b) in CAR_WINDOWS.items():
                    rec[f"CAR{k}_{bm}"] = wsum(p, a, b)
                pre = rec[f"CARpre[-5,-1]_{bm}"]
                for h in (5, 20):
                    rec[f"reversal+{h}_{bm}"] = (wsum(p, 0, h) / -pre) if pre < -0.02 else np.nan
                if bm == "SPY":
                    paths[(e.storm, tk)] = p
                    raw = ret[tk].iloc[e.i0 - 60:e.i0 + 61].values
                    rec["vol_pre[-60,-21]"] = ann_vol(raw[0:40])
                    rec["vol_post[0,+20]"] = ann_vol(raw[60:81])
                    rec["vol_late[+21,+60]"] = ann_vol(raw[81:121])
                    ar = p.values
                    rec["idio_pre"] = ann_vol(ar[0:40])
                    rec["idio_post"] = ann_vol(ar[60:81])
                    rec["idio_late"] = ann_vol(ar[81:121])
            if ok:
                rows.append(rec)
    return pd.DataFrame(rows), paths


# ---------------------------------------------------------------- statistics
def clustered_stats(x, clusters, nboot=5000):
    """Mean across events, event-cluster-robust SE, cluster bootstrap 95% CI."""
    x, c = np.asarray(x, float), np.asarray(clusters)
    m = ~np.isnan(x)
    x, c = x[m], c[m]
    n = len(x)
    if n == 0:
        return dict(n=0)
    mu = x.mean()
    uc = np.unique(c)
    g = len(uc)
    s = np.array([(x[c == k] - mu).sum() for k in uc])
    se = np.sqrt((s ** 2).sum()) / n * np.sqrt(g / (g - 1)) if g > 1 else np.nan
    boots = []
    for _ in range(nboot):
        pick = RNG.choice(uc, size=g, replace=True)
        boots.append(np.concatenate([x[c == k] for k in pick]).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=n, clusters=g, mean=mu, se=se, t=mu / se if se else np.nan,
                ci_lo=lo, ci_hi=hi, hit_pos=(x > 0).mean(), worst=x.min(), median=np.median(x))


def group_event_means(tt, cols):
    return (tt.groupby(["storm", "year", "period", "fl_landfall", "cluster", "group"], sort=False)[cols]
              .mean().reset_index())


def group_summary(gm, cols):
    out = []
    subsets = {
        "2020-24 all": gm.period == "2020-2024",
        "2020-24 FL landfall": (gm.period == "2020-2024") & gm.fl_landfall,
        "2020-24 non-FL landfall": (gm.period == "2020-2024") & ~gm.fl_landfall,
        "pre-2020 all": gm.period == "pre-2020",
    }
    for sname, sm in subsets.items():
        for grp in GROUPS:
            d = gm[sm & (gm.group == grp)]
            for col in cols:
                st = clustered_stats(d[col], d["cluster"])
                out.append(dict(subset=sname, group=grp, measure=col, **st))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- strategy
def basket_trade(i0, hold, members, extra_member_ret=None):
    e, x = i0 - 1, i0 - 1 + hold
    if x >= len(dates):
        return None
    r = [px[t].iloc[x] / px[t].iloc[e] - 1 for t in members
         if pd.notna(px[t].iloc[e]) and pd.notna(px[t].iloc[x])
         and market_model(t, "SPY", i0) is not None]          # same data filter as the event study
    if not r:
        return None
    if extra_member_ret is not None:
        r.append(extra_member_ret)
    b = np.mean(r)
    return dict(n_members=len(r), basket=b,
                ex_SPY=b - (px["SPY"].iloc[x] / px["SPY"].iloc[e] - 1),
                ex_KIE=b - (px["KIE"].iloc[x] / px["KIE"].iloc[e] - 1) if pd.notna(px["KIE"].iloc[e]) else np.nan,
                entry=dates[e].date(), exit=dates[x].date())


def strategy(ev, fnhc_ret=None):
    rows = []
    for _, e in ev.iterrows():
        for h in (5, 20):
            extra = fnhc_ret if (fnhc_ret is not None and e.storm == "Ian") else None
            tr = basket_trade(e.i0, h, GROUPS["FL"], extra)
            if tr:
                rows.append(dict(storm=e.storm, year=e.year, period=e.period, fl_landfall=e.fl_landfall,
                                 cluster=e.cluster, i0=e.i0, hold=h, **tr))
    return pd.DataFrame(rows)


def non_overlapping(st):
    keep = []
    for h, d in st.groupby("hold"):
        last_exit = -1
        for _, r in d.sort_values("i0").iterrows():
            if r.i0 - 1 >= last_exit:
                keep.append(r.name)
                last_exit = r.i0 - 1 + h
    return st.loc[keep]


def strategy_summary(st, label):
    out = []
    subsets = {"2020-24 all": st.period == "2020-2024",
               "2020-24 FL landfall": (st.period == "2020-2024") & st.fl_landfall,
               "pre-2020 FL landfall": (st.period == "pre-2020") & st.fl_landfall}
    for sname, sm in subsets.items():
        for h in (5, 20):
            d = st[sm & (st.hold == h)]
            for col in ("basket", "ex_SPY", "ex_KIE"):
                s = clustered_stats(d[col], d["cluster"])
                out.append(dict(run=label, subset=sname, hold=h, measure=col, **s))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- run
ev = build_events("primary")
ev_b = build_events("brief")
ev.drop(columns="i0").assign(T0_brief=ev_b["T0"]).to_csv(OUT / "events.csv", index=False)

tt, paths = event_ticker_table(ev)
tt.to_csv(OUT / "per_event_ticker.csv", index=False)

car_cols = [f"CAR{k}_{bm}" for bm in ("SPY", "KIE") for k in CAR_WINDOWS]
rev_cols = [f"reversal+{h}_SPY" for h in (5, 20)]
tt["dlogvol_post"] = np.log(tt["vol_post[0,+20]"] / tt["vol_pre[-60,-21]"])
tt["dlogvol_late"] = np.log(tt["vol_late[+21,+60]"] / tt["vol_pre[-60,-21]"])
tt["dlogidio_post"] = np.log(tt["idio_post"] / tt["idio_pre"])
tt["dlogidio_late"] = np.log(tt["idio_late"] / tt["idio_pre"])
vol_cols = ["dlogvol_post", "dlogvol_late", "dlogidio_post", "dlogidio_late"]

gm = group_event_means(tt, car_cols + vol_cols)
# reversal ratio on the GROUP-average path (share of the group's [-5,-1] drop recovered), only if drop > 2%
for h in (5, 20):
    pre = gm["CARpre[-5,-1]_SPY"]
    gm[f"reversal+{h}_SPY"] = np.where(pre < -0.02, gm[f"CAR[0,+{h}]_SPY"] / -pre, np.nan)
gm.to_csv(OUT / "per_event_group.csv", index=False)
group_summary(gm, car_cols + rev_cols + vol_cols).to_csv(OUT / "group_summary.csv", index=False)

# FL vs national vol difference per event ("relative to non-FL insurers")
piv = gm.pivot_table(index=["storm", "year", "period", "fl_landfall", "cluster"], columns="group",
                     values=vol_cols).reset_index()
vrel = []
for v in vol_cols:
    piv[(v, "FL-National")] = piv[(v, "FL")] - piv[(v, "National")]
    for sname, sm in {"2020-24 FL landfall": (piv.period == "2020-2024") & piv.fl_landfall,
                      "2020-24 non-FL landfall": (piv.period == "2020-2024") & ~piv.fl_landfall,
                      "pre-2020 FL landfall": (piv.period == "pre-2020") & piv.fl_landfall}.items():
        d = piv[sm]
        vrel.append(dict(subset=sname, measure=v, **clustered_stats(d[(v, "FL-National")], d["cluster"])))
pd.DataFrame(vrel).to_csv(OUT / "vol_FL_minus_national.csv", index=False)

# Strategy runs
runs = {
    "base (T0 = first close after landfall)": strategy(ev),
    "brief date convention": strategy(ev_b),
    "Ian + FNHC at -100%": strategy(ev, fnhc_ret=-1.0),
    "Ian + FNHC at -50%": strategy(ev, fnhc_ret=-0.5),
}
st_all = pd.concat([d.assign(run=k) for k, d in runs.items()])
st_all.to_csv(OUT / "strategy_trades.csv", index=False)
summ = [strategy_summary(d, k) for k, d in runs.items()]
summ.append(strategy_summary(non_overlapping(runs["base (T0 = first close after landfall)"]),
                             "base, non-overlapping trades only"))
pd.concat(summ).to_csv(OUT / "strategy_summary.csv", index=False)

# overshoot -> reversal: cross-section of FL tickers, OOS
fl = tt[(tt.group == "FL") & (tt.period == "2020-2024")].dropna(subset=["CARpre[-5,-1]_SPY", "CAR[0,+20]_SPY"])
def slope(d, y):
    return np.polyfit(d["CARpre[-5,-1]_SPY"], d[y], 1)[0]
bs = {y: [] for y in ("CAR[0,+5]_SPY", "CAR[0,+20]_SPY")}
cl_ids = fl.cluster.unique()
for _ in range(5000):
    pick = RNG.choice(cl_ids, len(cl_ids))
    d = pd.concat([fl[fl.cluster == k] for k in pick])
    for y in bs:
        bs[y].append(slope(d, y))
pd.DataFrame([dict(y=y, n=len(fl), slope=slope(fl, y), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5))
              for y, v in bs.items()]).to_csv(OUT / "overshoot_reversal_slope.csv", index=False)

# Milton sanity check vs brief (raw daily returns)
mil = ret.loc["2024-10-04":"2024-10-11", GROUPS["FL"]]
wk = px.loc["2024-10-04":"2024-10-11", GROUPS["FL"]]
(pd.concat([mil, (wk.iloc[-1] / wk.iloc[0] - 1).to_frame("week Fri4->Fri11").T])
   .to_csv(OUT / "milton_check.csv"))

# ---------------------------------------------------------------- charts
C = {"FL": "#2a78d6", "National": "#eb6834", "Reinsurer": "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "axes.spines.top": False,
                     "axes.spines.right": False})
REL = np.arange(-5, 61)


def cum_path(storm, tk):
    p = paths.get((storm, tk))
    if p is None:
        return None
    return np.concatenate([[0.0], p.loc[-5:60].cumsum().values])  # 0 at close T0-6


def group_path(storms, grp):
    ev_paths = []
    for s in storms:
        ps = [cum_path(s, t) for t in GROUPS[grp]]
        ps = [p for p in ps if p is not None and not np.isnan(p).any()]
        if ps:
            ev_paths.append(np.mean(ps, axis=0))
    return np.mean(ev_paths, axis=0) if ev_paths else None, len(ev_paths)


x = np.arange(-6, 61)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
panels = [("2020-24, FL landfalls", ev[(ev.period == "2020-2024") & ev.fl_landfall].storm),
          ("2020-24, other landfalls", ev[(ev.period == "2020-2024") & ~ev.fl_landfall].storm),
          ("Pre-2020 comparison (all)", ev[ev.period == "pre-2020"].storm)]
for ax, (title, storms) in zip(axes, panels):
    for grp in GROUPS:
        p, n = group_path(storms, grp)
        if p is not None:
            ax.plot(x, p * 100, color=C[grp], lw=2, label=f"{grp} (n={n})")
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.set_xlabel("Trading days relative to T0 (first close after landfall)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
axes[0].set_ylabel("Cumulative abnormal return vs SPY, %")
fig.suptitle("Average CAR path by insurer group (0 = close T0-6)", color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "car_paths_by_group.png", dpi=160)
plt.close(fig)

fl_storms = ev[ev.fl_landfall].storm.tolist()
fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), sharex=True)
for ax, s in zip(axes.flat, fl_storms):
    ps = [cum_path(s, t) for t in GROUPS["FL"]]
    for t, p in zip(GROUPS["FL"], ps):
        if p is not None:
            ax.plot(x, p * 100, color="#9a9994", lw=1)
    ok = [p for p in ps if p is not None and not np.isnan(p).any()]
    if ok:
        ax.plot(x, np.mean(ok, axis=0) * 100, color=C["FL"], lw=2.2)
    ax.axvline(0, color=INK2, lw=1, ls="--"); ax.axhline(0, color=INK2, lw=0.8)
    yr = int(ev.loc[ev.storm == s, "year"].iloc[0])
    ax.set_title(f"{s} {yr}  (n={len(ok)})", color=INK, loc="left", fontsize=10)
for ax in axes.flat[len(fl_storms):]:
    ax.axis("off")
fig.suptitle("FL-concentrated insurers, FL landfalls: CAR vs SPY in % (blue = basket mean, grey = single stocks)",
             color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "car_paths_fl_events.png", dpi=160)
plt.close(fig)

base = runs["base (T0 = first close after landfall)"]
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
for ax, h in zip(axes, (5, 20)):
    d = base[base.hold == h].reset_index(drop=True)
    cols = [C["FL"] if f else "#b9b8b2" for f in d.fl_landfall]
    ax.bar(range(len(d)), d.ex_SPY * 100, color=cols, width=0.7)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([f"{s} {str(y)[2:]}" for s, y in zip(d.storm, d.year)], rotation=70, fontsize=8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(6.5, color=INK2, lw=0.8, ls=":")
    ax.set_title(f"Buy FL basket at close T0-1, hold {h} days: excess vs SPY, %", color=INK, loc="left", fontsize=11)
axes[0].text(0.01, 0.97, "blue = FL landfall, grey = other landfall; left of dotted line = pre-2020",
             transform=axes[0].transAxes, color=INK2, fontsize=8.5, va="top")
fig.tight_layout()
fig.savefig(OUT / "strategy_per_event.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
for ax, y in zip(axes, ("CAR[0,+5]_SPY", "CAR[0,+20]_SPY")):
    for f, col, lab in ((True, C["FL"], "FL landfall"), (False, "#b9b8b2", "other landfall")):
        d = fl[fl.fl_landfall == f]
        ax.scatter(d["CARpre[-5,-1]_SPY"] * 100, d[y] * 100, s=40, color=col, edgecolor=SURF, lw=1.5, label=lab)
    ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Pre-landfall CAR [-5,-1] vs SPY, %"); ax.set_ylabel(f"Post CAR {y.split('_')[0][3:]}, %")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
fig.suptitle("FL insurers 2020-24: does a bigger pre-landfall drop reverse more?", color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "overshoot_vs_reversal.png", dpi=160)
plt.close(fig)

print("done:", len(tt), "event x ticker rows;", ev.cluster.nunique(), "clusters")
