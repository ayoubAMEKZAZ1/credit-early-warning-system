"""Point-in-time features and the 12-month default target.

One row per performing counterparty per month. Every feature at month t uses only information the bank
had at the end of month t (behaviour up to t, financial statements already published, the rating in
force). The target is default (90+ days past due) in the following 12 months, (t, t+12].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BEHAVIOUR = ["overdraft_utilisation", "util_change_6m", "months_util_above_95_6m", "limit_breach_days_3m",
             "payment_incidents_6m", "turnover_yoy_pct", "turnover_3m_vs_12m_pct", "days_past_due",
             "max_dpd_6m"]
RATING = ["rating", "rating_change_12m"]
FINANCIAL = ["net_debt_to_ebitda", "interest_coverage", "current_ratio", "revenue_growth_pct", "statement_age_months"]
CONTEXT = ["sector_util_change_6m"]
NUMERIC = BEHAVIOUR + RATING + FINANCIAL + CONTEXT
CATEGORICAL = ["segment"]
HORIZON = 12


def build_snapshots(D: dict) -> pd.DataFrame:
    P = D["monthly_panel"].merge(D["counterparties"][["counterparty_id", "segment", "sector", "region"]],
                                 on="counterparty_id")
    P["m"] = pd.PeriodIndex(P.month, freq="M")
    P = P.sort_values(["counterparty_id", "m"]).reset_index(drop=True)
    g = P.groupby("counterparty_id", sort=False)

    def roll(col, w, fn):
        return g[col].transform(lambda s: getattr(s.rolling(w, min_periods=1), fn)())

    P["util_change_6m"] = P.overdraft_utilisation - g.overdraft_utilisation.shift(6)
    P["months_util_above_95_6m"] = g.overdraft_utilisation.transform(lambda s: (s > 0.95).rolling(6, min_periods=1).sum())
    P["limit_breach_days_3m"] = roll("limit_breach_days", 3, "sum")
    P["payment_incidents_6m"] = roll("payment_incidents", 6, "sum")
    lt = np.log(P.account_turnover_mad_m)
    P["turnover_yoy_pct"] = 100 * (np.exp(lt - lt.groupby(P.counterparty_id).shift(12)) - 1)
    t3 = g.account_turnover_mad_m.transform(lambda s: s.rolling(3, min_periods=3).mean())
    t12 = g.account_turnover_mad_m.transform(lambda s: s.rolling(12, min_periods=12).mean())
    P["turnover_3m_vs_12m_pct"] = 100 * (t3 / t12 - 1)
    P["max_dpd_6m"] = roll("days_past_due", 6, "max")
    P["rating"] = pd.to_numeric(P.internal_rating, errors="coerce")
    P["rating_change_12m"] = P.rating - g.rating.shift(12)
    # sector conditions: average change in overdraft use across the sector this month
    P["sector_util_change_6m"] = P.groupby(["sector", "month"]).util_change_6m.transform("mean")

    # latest financial statements already published at month t
    F = D["financials"].copy()
    F["avail"] = pd.PeriodIndex(F.available_from, freq="M")
    F = F.sort_values("avail")
    P = P.sort_values("m")
    P["m_ts"] = P.m.dt.to_timestamp()
    F["avail_ts"] = F.avail.dt.to_timestamp()
    P = pd.merge_asof(P, F.drop(columns=["available_from", "avail"]), left_on="m_ts", right_on="avail_ts",
                      by="counterparty_id", direction="backward")
    P["statement_age_months"] = ((P.m_ts - P.avail_ts).dt.days / 30.4).round()

    # target: default within the next 12 months (performing rows only)
    truth = D["ground_truth"].set_index("counterparty_id").default_month
    dmon = pd.PeriodIndex(P.counterparty_id.map(truth).replace("", np.nan), freq="M")
    months_to_default = (dmon - P.m).map(lambda x: x.n if pd.notna(x) else np.nan) if len(P) else []
    P["months_to_default"] = np.asarray(months_to_default, dtype=float)
    P = P[P.status == "performing"].copy()
    P["target_12m"] = ((P.months_to_default >= 1) & (P.months_to_default <= HORIZON)).astype(int)
    # the current process: names already on watch (rating 8-9 or 30+ days past due)
    P["current_process_flag"] = ((P.rating >= 8) | (P.days_past_due >= 30)).astype(int)
    # target fully observable only if t + 12 is inside the data
    last = P.m.max()
    P["target_observable"] = (P.m + HORIZON) <= last
    return P.drop(columns=["m_ts", "avail_ts"]).sort_values(["m", "counterparty_id"]).reset_index(drop=True)
