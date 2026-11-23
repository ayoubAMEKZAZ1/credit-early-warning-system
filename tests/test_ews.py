import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ews import evaluate as E, models as M, portfolio as PF  # noqa: E402
from ews.features import NUMERIC, build_snapshots  # noqa: E402
from ews.generate import generate  # noqa: E402


@pytest.fixture(scope="module")
def data():
    D = generate(n=1500, seed=5)
    return D, build_snapshots(D)


def test_realistic_default_rate(data):
    D, _ = data
    P = D["monthly_panel"]
    years = D["ground_truth"].default_month.str[:4]
    perf = P[P.status == "performing"].groupby(P.month.str[:4]).counterparty_id.nunique()
    rate = years[years != ""].value_counts() / perf
    assert rate.between(0.01, 0.06).all()


def test_features_are_point_in_time(data):
    D, S = data
    F = D["financials"]
    # statements used at month t were published at or before t
    s = S.dropna(subset=["net_debt_to_ebitda"]).sample(300, random_state=0)
    for r in s.itertuples():
        pub = F[(F.counterparty_id == r.counterparty_id) & (F.available_from <= r.month)]
        assert len(pub) and np.isclose(pub.sort_values("available_from").net_debt_to_ebitda.iloc[-1], r.net_debt_to_ebitda)


def test_target_definition(data):
    _, S = data
    pos = S[S.target_12m == 1]
    assert pos.months_to_default.between(1, 12).all()
    assert (S.status == "performing").all()


def test_no_train_test_overlap(data):
    _, S = data
    tr, te = E.split(S)
    last_train_target = (pd.PeriodIndex(tr.month, freq="M").max() + 12).strftime("%Y-%m")
    assert last_train_target < te.month.min()


def test_models_and_capacity(data):
    _, S = data
    tr, _ = E.split(S)
    sc = E.score_all(S, M.fit_logistic(tr), M.fit_gbm(tr))
    _, te = E.split(sc)
    d = E.discrimination(te)
    assert (d.AUC > 0.6).all()
    a = E.alerts_at_capacity(te, "score_logistic")
    assert abs(a.groupby(te.month).mean().mean() - E.CAPACITY) < 0.005
    reasons = M.logistic_reasons(M.fit_logistic(tr), te.head(20))
    assert len(reasons) == 20 and all(isinstance(r, str) for r in reasons)


def test_portfolio_views(data):
    D, _ = data
    P = PF.enrich(D)
    m = PF.migration_matrix(P)
    assert np.allclose(m.sum(axis=1), 100, atol=0.5)
    diag = np.mean([m.loc[str(i), str(i)] for i in range(3, 7) if str(i) in m.index])
    assert 45 < diag < 80
    npl = PF.npl_ratio(P)
    assert (npl.iloc[-1] < 30).all()
