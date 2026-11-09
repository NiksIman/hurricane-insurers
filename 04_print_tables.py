"""Print the main result tables to the console (reads out/*.csv)."""
from pathlib import Path
import sys
import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
what = sys.argv[1] if len(sys.argv) > 1 else "all"
pct = lambda d, cs: d.assign(**{c: (d[c] * 100).round(1) for c in cs})

if what in ("milton", "all"):
    print((pd.read_csv(OUT / "milton_check.csv", index_col=0) * 100).round(1))

if what in ("events", "all"):
    g = pd.read_csv(OUT / "per_event_group.csv")
    cs = ["CARpre[-5,-1]_SPY", "CAR[0,+1]_SPY", "CAR[0,+5]_SPY", "CAR[0,+20]_SPY", "CAR[0,+60]_SPY",
          "CAR[0,+5]_KIE", "CAR[0,+20]_KIE"]
    print(pct(g[["storm", "year", "fl_landfall", "group"] + cs], cs).to_string(index=False))

if what in ("summary", "all"):
    s = pd.read_csv(OUT / "group_summary.csv")
    s = s[s.measure.str.contains("SPY") | s.measure.str.startswith("dlog")]
    keep = ["subset", "group", "measure", "n", "clusters", "mean", "se", "t", "ci_lo", "ci_hi", "hit_pos", "worst"]
    s = s[keep]
    for c in ["mean", "se", "ci_lo", "ci_hi", "worst"]:
        s[c] = (s[c] * 100).round(1)
    s["t"] = s["t"].round(2); s["hit_pos"] = s["hit_pos"].round(2)
    print(s.to_string(index=False))

if what in ("vol", "all"):
    v = pd.read_csv(OUT / "vol_FL_minus_national.csv")
    print(v.round(3).to_string(index=False))

if what in ("strategy", "all"):
    s = pd.read_csv(OUT / "strategy_summary.csv")
    keep = ["run", "subset", "hold", "measure", "n", "clusters", "mean", "median", "se", "t", "ci_lo", "ci_hi", "hit_pos", "worst"]
    s = s[keep]
    for c in ["mean", "median", "se", "ci_lo", "ci_hi", "worst"]:
        s[c] = (s[c] * 100).round(1)
    s["t"] = s["t"].round(2); s["hit_pos"] = s["hit_pos"].round(2)
    print(s.to_string(index=False))

if what in ("trades", "all"):
    t = pd.read_csv(OUT / "strategy_trades.csv")
    t = t[t.run == "base (T0 = first close after landfall)"]
    print(pct(t[["storm", "year", "fl_landfall", "hold", "entry", "exit", "n_members", "basket", "ex_SPY", "ex_KIE"]],
              ["basket", "ex_SPY", "ex_KIE"]).to_string(index=False))

if what in ("slope", "all"):
    print(pd.read_csv(OUT / "overshoot_reversal_slope.csv").round(3))

if what in ("fltickers", "all"):
    t = pd.read_csv(OUT / "per_event_ticker.csv")
    t = t[(t.group == "FL") & t.fl_landfall]
    cs = ["CARpre[-5,-1]_SPY", "CAR[0,+1]_SPY", "CAR[0,+5]_SPY", "CAR[0,+20]_SPY", "CAR[0,+60]_SPY"]
    d = pct(t[["storm", "year", "ticker"] + cs + ["reversal+5_SPY", "reversal+20_SPY",
             "vol_pre[-60,-21]", "vol_post[0,+20]", "vol_late[+21,+60]"]], cs)
    for c in ["reversal+5_SPY", "reversal+20_SPY", "vol_pre[-60,-21]", "vol_post[0,+20]", "vol_late[+21,+60]"]:
        d[c] = d[c].round(2)
    print(d.to_string(index=False))
