#!/usr/bin/env python3
# ------------------------------------------------------------------
# scrape_palisades_fire.py   –   STAT‑3215Q regression data builder
#
# Usage:  python scrape_palisades_fire.py
# Requires: requests, beautifulsoup4, pandas  (pip install …)
# ------------------------------------------------------------------

import requests, re, csv, time, datetime as dt
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL   = "https://www.fire.ca.gov"
INDEX_URL  = "https://www.fire.ca.gov/incidents/2025/1/7/palisades-fire/updates"
HEADERS    = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/125.0 Safari/537.36")
}

# ------------------------------------------------------------------
# 1. Collect every bulletin URL from the index page
# ------------------------------------------------------------------
def get_update_links(index_url: str) -> list[str]:
    html = requests.get(index_url, headers=HEADERS, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")

    links = []
    for a in soup.select('a[href*="updates/"]'):
        href = a["href"]
        # make absolute
        if href.startswith("/"):
            href = BASE_URL + href
        elif not href.startswith("http"):
            href = BASE_URL + "/" + href.lstrip("/")
        if href not in links:
            links.append(href)
    return links


# ------------------------------------------------------------------
# 2. Parse one bulletin page → dict of fields we care about
# ------------------------------------------------------------------
size_re   = re.compile(r"\bsize\s*([\d,]+)", re.I)
cont_re   = re.compile(r"\bcontainment\s*([\d]+)%", re.I)
res_pat   = lambda label: re.compile(fr"(\d+)\s+{label}", re.I)

dt_long   = re.compile(r"([A-Za-z]+ \d{1,2}, \d{4} at \d{1,2}:\d{2} [AP]M)", re.I)
dt_slash  = re.compile(r"(\d{1,2}/\d{1,2}/\d{4} at \d{1,2}:\d{2} [AP]M)", re.I)

def parse_bulletin(url: str) -> dict | None:
    html  = requests.get(url, headers=HEADERS, timeout=30).text
    soup  = BeautifulSoup(html, "html.parser")
    head  = soup.find("h1").get_text(" ", strip=True).lower()
    text  = soup.get_text(" ", strip=True)

    # ---- timestamp (skip pages with no date) ----
    m = dt_long.search(head) or dt_slash.search(head)
    if not m:
        return None
    fmt = "%B %d, %Y at %I:%M %p" if "," in m.group(1) else "%m/%d/%Y at %I:%M %p"
    ts  = dt.datetime.strptime(m.group(1), fmt)

    # ---- numeric fields ----
    size   = size_re.search(text)
    cont   = cont_re.search(text)
    engines      = res_pat("engines?").search(text)
    hand_crews   = res_pat("hand crews?").search(text)
    tenders      = res_pat("water tenders?").search(text)
    aircraft     = res_pat("(helicopters|aircraft)").search(text)
    personnel    = res_pat("personnel").search(text)
    threatened   = res_pat("structures? threatened").search(text)
    destroyed    = res_pat("structures? destroyed").search(text)

    return {
        "datetime"        : ts,
        "acres_burned"    : int(size.group(1).replace(",", "")) if size else None,
        "containment_pct" : int(cont.group(1))                  if cont else None,
        "engines"         : int(engines.group(1))               if engines else None,
        "hand_crews"      : int(hand_crews.group(1))            if hand_crews else None,
        "water_tenders"   : int(tenders.group(1))               if tenders else None,
        "aircraft"        : int(aircraft.group(1))              if aircraft else None,
        "personnel"       : int(personnel.group(1))             if personnel else None,
        "structures_threatened": int(threatened.group(1))       if threatened else None,
        "structures_destroyed" : int(destroyed.group(1))        if destroyed  else None,
        "url"             : url
    }


# ------------------------------------------------------------------
# 3. Main routine – scrape, save CSVs
# ------------------------------------------------------------------
def main():
    links = get_update_links(INDEX_URL)
    print(f"Discovered {len(links)} bulletin pages")

    rows = []
    for i, url in enumerate(links, 1):
        rec = parse_bulletin(url)
        if rec:
            rows.append(rec)
        time.sleep(0.3)        # polite crawl
        if i % 50 == 0:
            print(f"  …parsed {i} / {len(links)}")

    if not rows:
        raise RuntimeError("No rows parsed – check regex or connection.")

    # FULL timeline CSV
    rows.sort(key=lambda r: r["datetime"])
    full_csv = "palisades_updates_full.csv"
    with open(full_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"✅  Wrote {full_csv}  ({len(rows)} rows)")

    # DAILY snapshot (last bulletin of each day)
    df = pd.DataFrame(rows)
    df["date"] = df["datetime"].dt.date
    daily = (df.sort_values("datetime")
               .groupby("date").last()
               .reset_index(drop=False)
               .rename(columns={"date": "calendar_date"}))
    daily_csv = "palisades_daily.csv"
    daily.to_csv(daily_csv, index=False)
    print(f"✅  Wrote {daily_csv}  ({len(daily)} rows)")


if __name__ == "__main__":
    main()
