"""Cache Wikidata P571 values for churches lacking WikiChurches built_year."""
import datetime as dt
import json
import time
from pathlib import Path

import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathing import ROOT, RAW as RAW_DIR, PROCESSED
RAW = RAW_DIR / "churches.json"
CACHE = PROCESSED / "wikidata_cache.json"
ENDPOINT = "https://query.wikidata.org/sparql"


def main():
    churches = json.loads(RAW.read_text(encoding="utf-8"))
    missing = sorted(q for q, c in churches.items() if c.get("built_year") is None)
    if CACHE.exists():
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        cache = {"endpoint": ENDPOINT, "queried_at_utc": None, "records": {}, "errors": []}
    pending = [q for q in missing if q not in cache["records"]]
    session = requests.Session()
    session.headers.update({"User-Agent": "WikiChurches-research/1.0 (academic data preparation)", "Accept": "application/sparql-results+json"})
    for start in range(0, len(pending), 100):
        batch = pending[start:start + 100]
        values = " ".join("wd:" + q for q in batch)
        query = f"SELECT ?item ?date WHERE {{ VALUES ?item {{ {values} }} OPTIONAL {{ ?item wdt:P571 ?date }} }}"
        for attempt in range(5):
            try:
                response = session.get(ENDPOINT, params={"query": query}, timeout=90)
                response.raise_for_status()
                bindings = response.json()["results"]["bindings"]
                break
            except (requests.RequestException, ValueError, KeyError) as exc:
                if attempt == 4:
                    cache["errors"].append({"batch_start": start, "qids": batch, "error": str(exc)})
                    bindings = None
                else:
                    time.sleep(min(60, 2 ** attempt * 3))
        if bindings is not None:
            found = {q: [] for q in batch}
            for row in bindings:
                qid = row["item"]["value"].rsplit("/", 1)[-1]
                if qid in found and "date" in row:
                    found[qid].append(row["date"]["value"])
            for qid, dates in found.items():
                years = sorted(set(int(value[:4]) for value in dates if len(value) >= 4 and value[:4].isdigit()))
                cache["records"][qid] = {"p571_values": sorted(set(dates)), "year": years[0] if len(years) == 1 else None, "ambiguous": len(years) > 1}
        cache["queried_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wikidata: {min(start + 100, len(pending))}/{len(pending)} pending QIDs; cached={len(cache['records'])}", flush=True)
        time.sleep(1)
    print(f"Complete: missing={len(missing)}, cached={len(cache['records'])}, errors={len(cache['errors'])}")


if __name__ == "__main__":
    main()
