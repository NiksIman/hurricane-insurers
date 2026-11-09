"""Tier 1, step 2: verify landfall dates/times against NHC HURDAT2 (official best track).

HURDAT2 marks landfalls with record identifier 'L' (time in UTC). We pull every 'L' record
for the brief's storms and keep only those over the US mainland (rough lat/lon box).
Output: data/hurdat2_landfalls.csv (all US 'L' records for these storms).
"""
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-092326.txt"
LOCAL = ROOT / "data" / "raw" / "hurdat2-1851-2025-092326.txt"

STORMS = {  # (year, name) as in HURDAT2 headers
    (2005, "KATRINA"), (2008, "IKE"), (2011, "IRENE"), (2012, "SANDY"), (2017, "HARVEY"),
    (2017, "IRMA"), (2018, "MICHAEL"),
    (2020, "HANNA"), (2020, "ISAIAS"), (2020, "LAURA"), (2020, "SALLY"), (2020, "DELTA"),
    (2020, "ZETA"), (2021, "IDA"), (2021, "NICHOLAS"), (2022, "IAN"), (2022, "NICOLE"),
    (2023, "IDALIA"), (2024, "BERYL"), (2024, "DEBBY"), (2024, "FRANCINE"), (2024, "HELENE"),
    (2024, "MILTON"),
}

if not LOCAL.exists():
    urllib.request.urlretrieve(URL, LOCAL)

def parse_lat(s):
    v = float(s[:-1]); return v if s[-1] == "N" else -v

def parse_lon(s):
    v = float(s[:-1]); return -v if s[-1] == "W" else v

rows, cur = [], None
for line in LOCAL.read_text().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if line.startswith("AL") and len(parts) >= 3 and parts[2].isdigit():
        cur = (int(parts[0][4:8]), parts[1])
        continue
    if cur in STORMS and len(parts) > 3 and parts[2] == "L":
        lat, lon = parse_lat(parts[4]), parse_lon(parts[5])
        us = 24.0 <= lat <= 46.0 and -98.0 <= lon <= -66.0   # US Gulf/East coast box
        rows.append(dict(year=cur[0], storm=cur[1], date=parts[0], time_utc=parts[1],
                         status=parts[3], lat=lat, lon=lon, wind_kt=int(parts[6]), us_box=us))

import csv
with open(ROOT / "data" / "hurdat2_landfalls.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows:
    if r["us_box"]:
        print(f"{r['year']} {r['storm']:9s} {r['date']} {r['time_utc']}Z {r['status']} "
              f"{r['lat']:5.1f} {r['lon']:6.1f} {r['wind_kt']:3d}kt")
found = {(r["year"], r["storm"]) for r in rows}
print("missing:", sorted(STORMS - found))
