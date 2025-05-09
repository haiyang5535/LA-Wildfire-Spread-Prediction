import requests, re, csv, datetime as dt
from bs4 import BeautifulSoup

base = "https://www.fire.ca.gov"
index = "https://www.fire.ca.gov/incidents/2025/1/7/palisades-fire/updates"
rows  = []

# 1. Grab every update link
soup = BeautifulSoup(requests.get(index).text, "html.parser")
links = [base + a["href"] for a in soup.select("a") if "/updates/" in a["href"]]

for url in links:
    page  = BeautifulSoup(requests.get(url).text, "html.parser")
    head  = page.find("h1").text.strip()                   # e.g. "Update as of January 15, 2025 at 5:54 PM"
    ts    = dt.datetime.strptime(head[13:], "%B %d, %Y at %I:%M %p")

    body  = page.get_text(" ").lower()

    acres = re.search(r"(\d[\d,]*) acres", body)
    cont  = re.search(r"(\d+)% containment", body)

    engines = re.search(r"(\d+) engines", body)
    crews   = re.search(r"(\d+) hand crew", body)

    rows.append({
        "datetime"      : ts,
        "acres_burned"  : int(acres.group(1).replace(",", "")) if acres else None,
        "containment"   : int(cont.group(1)) if cont else None,
        "engines"       : int(engines.group(1)) if engines else None,
        "hand_crews"    : int(crews.group(1)) if crews else None,
        "url"           : url
    })

# 3. Write the CSV
with open("palisades_updates_full.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
