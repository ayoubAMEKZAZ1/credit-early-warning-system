"""Early-warning scores, from the simplest to the most flexible.

current_process : what many banks run today: names with rating 8-9 or 30+ days past due go on watch
expert_rules    : a points scorecard of behaviour triggers an analyst would write down
logistic        : interpretable scorecard; each alert comes with the factors that drove it
gbm             : gradient boosting (non-linear benchmark)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL, NUMERIC

NAMES = {
    "overdraft_utilisation": "overdraft use", "util_change_6m": "6-month change in overdraft use",
    "months_util_above_95_6m": "months with overdraft near limit", "limit_breach_days_3m": "limit-breach days",
    "payment_incidents_6m": "payment incidents", "turnover_yoy_pct": "account turnover vs a year ago",
    "turnover_3m_vs_12m_pct": "turnover vs its 12-month average", "days_past_due": "days past due",
    "max_dpd_6m": "worst arrears in 6 months", "rating": "internal rating (9 = worst)",
    "rating_change_12m": "rating change over 12 months", "net_debt_to_ebitda": "net debt / EBITDA",
    "interest_coverage": "interest coverage", "current_ratio": "current ratio",
    "revenue_growth_pct": "revenue growth", "statement_age_months": "age of financial statements",
    "sector_util_change_6m": "sector-wide change in overdraft use",
}

RULES = [  # (description, condition, points)
    ("Overdraft use above 90%", lambda d: d.overdraft_utilisation > 0.90, 1),
    ("Overdraft above 95% for 3+ of the last 6 months", lambda d: d.months_util_above_95_6m >= 3, 1),
    ("10+ limit-breach days in 3 months", lambda d: d.limit_breach_days_3m >= 10, 1),
    ("2+ payment incidents in 6 months", lambda d: d.payment_incidents_6m >= 2, 1),
    ("Account turnover down 15%+ on a year earlier", lambda d: d.turnover_yoy_pct <= -15, 1),
    ("Any arrears of 15+ days in 6 months", lambda d: d.max_dpd_6m >= 15, 2),
    ("Net debt above 5x EBITDA", lambda d: d.net_debt_to_ebitda > 5, 1),
    ("Interest coverage below 1.5x", lambda d: d.interest_coverage < 1.5, 1),
    ("Internal rating 7 or worse", lambda d: d.rating >= 7, 1),
]


def score_current_process(d):
    """Rank names on watch first (by rating, then arrears); everyone else by rating."""
    return d.current_process_flag * 100 + d.rating + d.days_past_due / 100


def score_expert_rules(d):
    pts = sum(np.where(cond(d).fillna(False), p, 0) for _, cond, p in RULES)
    return pts + d.overdraft_utilisation.fillna(0) / 10   # tie-break by overdraft use


def _prep():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL)])


def fit_logistic(train):
    m = Pipeline([("prep", _prep()), ("lr", LogisticRegression(C=0.3, max_iter=2000))])
    return m.fit(train[NUMERIC + CATEGORICAL], train.target_12m)


def fit_gbm(train, seed=0):
    X = train[NUMERIC].copy()
    X["segment_code"] = train.segment.astype("category").cat.codes
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=150,
                                       l2_regularization=1.0, random_state=seed)
    m.fit(X, train.target_12m)
    m.segment_categories_ = list(train.segment.astype("category").cat.categories)
    return m


def predict_gbm(m, d):
    X = d[NUMERIC].copy()
    X["segment_code"] = pd.Categorical(d.segment, categories=m.segment_categories_).codes
    return m.predict_proba(X)[:, 1]


def logistic_reasons(m, d, top=3):
    """Top factors behind each logistic score: coefficient x standardised value (positive = raises risk)."""
    prep, lr = m.named_steps["prep"], m.named_steps["lr"]
    Z = prep.transform(d[NUMERIC + CATEGORICAL])
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    contrib = Z[:, :len(NUMERIC)] * lr.coef_[0][:len(NUMERIC)]
    Zn = Z[:, :len(NUMERIC)]
    out = []
    for row, z in zip(contrib, Zn):
        idx = np.argsort(-row)[:top]
        out.append("; ".join(f"{NAMES[NUMERIC[i]]} {'high' if z[i] > 0 else 'low'}" for i in idx if row[i] > 0.05))
    return out


def logistic_coefficients(m):
    lr = m.named_steps["lr"]
    return pd.Series(lr.coef_[0][:len(NUMERIC)], index=[NAMES[c] for c in NUMERIC]).sort_values()
