"""Out-of-time evaluation of the early-warning scores.

Train : monthly snapshots 2021-07 to 2022-12 (12-month targets end by 2023-12, so nothing from the test
        period leaks into training)
Test  : monthly snapshots 2024-01 to 2024-12 (targets observed up to 2025-12)
Lead time : defaults during 2025, scored monthly from 2024-01 to 2025-11

Capacity: risk teams can review only a fixed number of names. Each month the top 4% of the performing
book by score is "alerted" (about the volume the current process flags); precision and recall are measured at that capacity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import models as M

TRAIN = ("2021-07", "2022-12")
TEST = ("2024-01", "2024-12")
CAPACITY = 0.04   # about the share of the book the current process puts on watch


def split(S):
    tr = S[(S.month >= TRAIN[0]) & (S.month <= TRAIN[1])]
    te = S[(S.month >= TEST[0]) & (S.month <= TEST[1])]
    return tr, te


def ks_stat(y, s):
    o = np.argsort(-s)
    y = np.asarray(y)[o]
    cum_pos = np.cumsum(y) / y.sum()
    cum_neg = np.cumsum(1 - y) / (1 - y).sum()
    return float(np.max(np.abs(cum_pos - cum_neg)))


def alerts_at_capacity(d, score_col, capacity=CAPACITY):
    """Flag the top `capacity` share of each month's performing book."""
    r = d.groupby("month")[score_col].rank(ascending=False, method="first")
    n = d.groupby("month")[score_col].transform("size")
    return (r <= np.ceil(capacity * n)).astype(int)


def psi(a, b, bins=10):
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a)
    pb = np.histogram(b, edges)[0] / len(b)
    pa, pb = np.clip(pa, 1e-6, None), np.clip(pb, 1e-6, None)
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def score_all(S, logit, gbm):
    S = S.copy()
    S["score_current_process"] = M.score_current_process(S)
    S["score_expert_rules"] = M.score_expert_rules(S)
    S["score_logistic"] = logit.predict_proba(S[M.NUMERIC + M.CATEGORICAL])[:, 1]
    S["score_gbm"] = M.predict_gbm(gbm, S)
    return S


MODELS = {"Current process (rating 8-9 or 30+ DPD)": "score_current_process", "Expert rules": "score_expert_rules",
          "Logistic scorecard": "score_logistic", "Gradient boosting": "score_gbm"}


def discrimination(te):
    rows = []
    for name, col in MODELS.items():
        y, s = te.target_12m.values, te[col].values
        a = alerts_at_capacity(te, col)
        rows.append({"model": name, "AUC": roc_auc_score(y, s), "Gini": 2 * roc_auc_score(y, s) - 1,
                     "KS": ks_stat(y, s), "PR_AUC": average_precision_score(y, s),
                     "precision_at_capacity": te.target_12m[a == 1].mean(),
                     "recall_at_capacity": te.target_12m[a == 1].sum() / te.target_12m.sum(),
                     "Brier": brier_score_loss(y, s) if col in ("score_logistic", "score_gbm") else np.nan})
    return pd.DataFrame(rows)


def lead_time(scored, truth):
    """For defaults in 2025: share flagged at least once in the 12 months before default, and how early."""
    d = scored[(scored.month >= "2024-01") & (scored.month <= "2025-11")].copy()
    dm = truth.set_index("counterparty_id").default_month
    defaulters = dm[(dm >= "2025-01") & (dm <= "2025-12")]
    rows = []
    for name, col in MODELS.items():
        flag = (d.current_process_flag == 1) if col == "score_current_process" else alerts_at_capacity(d, col) == 1
        f = d[flag & d.counterparty_id.isin(defaulters.index)]
        leads = []
        for cp, dmon in defaulters.items():
            fm = f[f.counterparty_id == cp].month
            win = fm[(fm < dmon) & (fm >= str(pd.Period(dmon, "M") - 12))]
            leads.append((pd.Period(dmon, "M") - pd.Period(win.min(), "M")).n if len(win) else np.nan)
        L = pd.Series(leads)
        rows.append({"model": name, "defaults": len(L), "flagged_before_default_pct": 100 * L.notna().mean(),
                     "flagged_6m_or_more_ahead_pct": 100 * (L >= 6).mean(),
                     "median_lead_months": L.median(), "alerts_per_month": flag.groupby(d.month).sum().mean()})
    return pd.DataFrame(rows)


def calibration_table(te, col="score_logistic", bins=10):
    q = pd.qcut(te[col], bins, labels=False, duplicates="drop")
    return te.groupby(q).agg(mean_predicted=(col, "mean"), observed_rate=("target_12m", "mean"), n=(col, "size"))
