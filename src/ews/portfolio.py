"""Portfolio-quality views: exposure mix, NPL ratio, rating migration and a cost-of-risk proxy.

Cost of risk (proxy): provisions = performing exposure x PD(rating) x LGD + defaulted exposure x LGD.
Annual cost of risk = (provisions at year end - provisions at the start of the year + provisions released
by write-offs) / average exposure. It is a simplified expected-loss view, not an IFRS 9 calculation
(see the separate IFRS 9 reconciliation project for staging and lifetime ECL).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .generate import LGD, PD_BY_RATING

RATING_BUCKETS = {**{r: "1-3 (strong)" for r in (1, 2, 3)}, **{r: "4-6 (satisfactory)" for r in (4, 5, 6)},
                  7: "7 (weak)", 8: "8-9 (watch)", 9: "8-9 (watch)"}
BUCKET_ORDER = ["1-3 (strong)", "4-6 (satisfactory)", "7 (weak)", "8-9 (watch)", "Default"]


def enrich(D):
    P = D["monthly_panel"].merge(D["counterparties"][["counterparty_id", "segment", "sector", "region"]],
                                 on="counterparty_id")
    r = pd.to_numeric(P.internal_rating, errors="coerce")
    P["rating_bucket"] = r.map(RATING_BUCKETS).fillna("Default")
    P["provision_mad_m"] = np.where(P.status == "default", P.exposure_mad_m * LGD,
                                    P.exposure_mad_m * r.map(PD_BY_RATING).fillna(1) * LGD)
    P["npl_exposure"] = np.where(P.status == "default", P.exposure_mad_m, 0.0)
    return P


def npl_ratio(P, by="segment"):
    g = P.groupby(["month", by])[["npl_exposure", "exposure_mad_m"]].sum()
    return (100 * g.npl_exposure / g.exposure_mad_m).unstack(by)


def exposure_mix(P, month, by):
    d = P[P.month == month]
    t = d.pivot_table(index=by, columns="rating_bucket", values="exposure_mad_m", aggfunc="sum", fill_value=0)
    return t.reindex(columns=[b for b in BUCKET_ORDER if b in t.columns])


def migration_matrix(P, start="2024-12", end="2025-12", min_names=10):
    a = P[P.month == start].set_index("counterparty_id").internal_rating
    b = P[P.month == end].set_index("counterparty_id").internal_rating
    j = pd.concat([a.rename("from"), b.rename("to")], axis=1).dropna()
    j = j[j["from"] != "D"]
    counts = j["from"].value_counts()
    j = j[j["from"].map(counts) >= min_names]          # grades with too few names are not shown
    m = pd.crosstab(j["from"], j["to"], normalize="index") * 100
    order = [str(i) for i in range(1, 10)] + ["D"]
    rows = [r for r in order[:-1] if r in m.index]
    return m.reindex(index=rows, columns=order, fill_value=0).round(1)


def cost_of_risk(P, by=None):
    """Annual cost of risk (% of average exposure), overall or by a dimension."""
    keys = ["year"] + ([by] if by else [])
    P = P.assign(year=P.month.str[:4])
    dec = P[P.month.str.endswith("-12")]
    prov = dec.groupby(keys).provision_mad_m.sum()
    avg_exp = P.groupby(keys + ["month"]).exposure_mad_m.sum().groupby(keys).mean()
    # write-offs: defaulted names leaving the book during the year release their full provision
    last_seen = P.groupby("counterparty_id").month.max()
    wo = P[(P.status == "default") & (P.month == P.counterparty_id.map(last_seen)) & (P.month < "2025-12")]
    wo = wo.assign(year=wo.month.str[:4]).groupby(keys).provision_mad_m.sum()
    rows = []
    for k in prov.index:
        y = k[0] if by else k
        prev = (str(int(y) - 1),) + (k[1:] if by else ())
        prev = prev if by else prev[0]
        if prev not in prov.index:
            continue
        cor = (prov[k] - prov[prev] + wo.get(k, 0.0)) / avg_exp[k] * 100
        rows.append({**({"year": k[0], by: k[1]} if by else {"year": k}), "cost_of_risk_pct": cor})
    return pd.DataFrame(rows)
