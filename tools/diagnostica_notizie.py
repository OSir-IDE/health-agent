"""Diagnostica del giornale sui feed reali: stato delle fonti e taratura delle soglie.

Uso:  python tools/diagnostica_notizie.py
Stampa solo titoli pubblici delle notizie, nessun dato personale: l'output si può condividere.
"""
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

load_dotenv(override=True)
from health_agent.news import CONTENT, cluster, fetch_all, similarity, vectorize

now = datetime.now(timezone.utc)
sources = json.loads((CONTENT / "news_sources.json").read_text(encoding="utf-8"))
items, failed, status = fetch_all(sources, now)

print("== FONTI ==")
print(f"  {'fonte':<20} {'articoli':>8}  ultimo articolo   stato")
for s in sources:
    st = status[s["name"]]
    newest = st["newest"].astimezone().strftime("%d/%m %H:%M") if st["newest"] else "—"
    print(f"  {s['name']:<20} {len(st['items']):>8}  {newest:<16}  {st['error'] or 'ok'}")

vectors, mode, warn = vectorize(items)
print(f"\n== MODO: {mode} ==" + (f"  ({warn})" if warn else ""))
sim = similarity(items, vectors)
testata = lambda i: items[i]["source"].split()[0]

grid, band = ((0.60, 0.65, 0.70, 0.75, 0.80), (0.50, 0.85)) if mode == "semantico" else \
             ((0.15, 0.18, 0.21, 0.25, 0.30), (0.12, 0.35))
print("\n== STORIE SU PIU' TESTATE, AL VARIARE DELLA SOGLIA ==")
for thr in grid:
    groups = cluster(items, sim, thr)
    sizes = [len({testata(i) for i in g}) for g in groups]
    multi_it = sum(1 for g in groups if len({testata(i) for i in g if items[i]["lang"] == "it"}) >= 2)
    mixed = sum(1 for g in groups if len({items[i]["lang"] for i in g}) == 2)
    print(f"  soglia {thr:.2f}: {sum(1 for x in sizes if x >= 2):3d} su 2+ testate · {multi_it:3d} su 2+ testate italiane"
          f" · {mixed:3d} in due lingue · max testate per storia {max(sizes, default=0)}")

pairs = [(sim[a, b], a, b) for a in range(len(items)) for b in range(a + 1, len(items))
         if testata(a) != testata(b) and band[0] <= sim[a, b] < band[1]]
pairs.sort(reverse=True)
step = max(1, len(pairs) // 30)
sample = pairs[::step][:30]
print(f"\n== 30 COPPIE DISTRIBUITE NELLA FASCIA INCERTA {band[0]:.2f}-{band[1]:.2f} ({len(pairs)} coppie in tutto) ==")
print("   Per ognuna: è la stessa notizia (S) o no (N)?")
for k, (v, a, b) in enumerate(sample, 1):
    print(f"\n  {k:2d}. {v:.2f}  [{items[a]['source']}] {items[a]['title'][:90]}")
    print(f"            [{items[b]['source']}] {items[b]['title'][:90]}")
