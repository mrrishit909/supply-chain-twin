"""eta-forecast-service and supplier delay risk: gradient boosting against the obvious baselines.

History rows are (etd day, supplier, port, plant, carrier, month, congestion, base days, promised days, actual days, tier, allocation).
Port -1 means an overland lane.
"""
import pickle

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score

ETA_VERSION, RISK_VERSION = "eta-gb-1.0.0", "supplier-delay-gb-1.0.0"
LATE_DAYS = 3
TEST_DAYS = 180           # the most recent half year is held out
CATS = [0, 1, 2, 3]       # port, plant, carrier, month


def eta_features(port, plant, carrier, month, congestion, base):
    return [port + 1, plant, carrier, month, congestion, base]


def _split(history, max_train=120_000, seed=0):
    h = np.array(history, float)
    x = np.column_stack([h[:, 2] + 1, h[:, 3], h[:, 4], h[:, 5], h[:, 6], h[:, 7]])
    test = h[:, 0] > -TEST_DAYS
    tr = np.flatnonzero(~test)
    if len(tr) > max_train:
        tr = np.random.default_rng(seed).choice(tr, max_train, replace=False)
    return h, x, tr, np.flatnonzero(test)


def train_eta(history):
    """-> (models {p10, p50, p90}, metrics). Baselines: the promised transit time, and the lane's median in the training period."""
    h, x, tr, te = _split(history)
    y = h[:, 9]
    models = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200, learning_rate=0.1, categorical_features=CATS, random_state=0).fit(x[tr], y[tr])
              for q in (0.1, 0.5, 0.9)}
    pred = {q: m.predict(x[te]) for q, m in models.items()}
    lane = {}
    for i in tr:
        lane.setdefault((h[i, 2], h[i, 3]), []).append(y[i])
    med = {k: float(np.median(v)) for k, v in lane.items()}
    overall = float(np.median(y[tr]))
    lane_pred = np.array([med.get((h[i, 2], h[i, 3]), overall) for i in te])
    metrics = {"n_train": int(len(tr)), "n_test": int(len(te)),
               "model": {"mae_days": round(mean_absolute_error(y[te], pred[0.5]), 2), "interval_80_coverage": round(float(((y[te] >= pred[0.1]) & (y[te] <= pred[0.9])).mean()), 3),
                         "interval_80_width_days": round(float((pred[0.9] - pred[0.1]).mean()), 2)},
               "baseline": {"name": "lane median transit time", "mae_days": round(mean_absolute_error(y[te], lane_pred), 2),
                            "promised_date_mae_days": round(mean_absolute_error(y[te], h[te, 8]), 2)}}
    return models, metrics


def train_risk(history):
    """Probability that a shipment arrives more than LATE_DAYS after its promised date; calibrated. Baseline: the overall late rate."""
    h, x, tr, te = _split(history)
    late = (h[:, 9] - h[:, 8] > LATE_DAYS).astype(int)
    sup_rate = {}
    for i in tr:
        sup_rate.setdefault(h[i, 1], []).append(late[i])
    prior = float(late[tr].mean())
    rate = {k: (sum(v) + 5 * prior) / (len(v) + 5) for k, v in sup_rate.items()}        # shrink small suppliers toward the overall rate
    xs = np.column_stack([x, [rate.get(s, prior) for s in h[:, 1]]])
    clf = CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, categorical_features=CATS, random_state=0), method="isotonic", cv=3).fit(xs[tr], late[tr])
    p = clf.predict_proba(xs[te])[:, 1]
    metrics = {"n_train": int(len(tr)), "n_test": int(len(te)), "late_rate_test": round(float(late[te].mean()), 3),
               "model": {"auc": round(roc_auc_score(late[te], p), 3), "brier": round(brier_score_loss(late[te], p), 4)},
               "baseline": {"name": "overall late rate", "auc": 0.5, "brier": round(brier_score_loss(late[te], np.full(len(te), prior)), 4)}}
    # supplier risk score = mean predicted late probability over the supplier's held-out shipments (falls back to its shrunk history)
    score = {}
    for s in set(h[:, 1]):
        sel = h[te, 1] == s
        score[int(s)] = round(float(p[sel].mean()) if sel.any() else rate.get(s, prior), 4)
    return clf, metrics, score, rate


def dumps(obj):
    return pickle.dumps(obj)


def loads(blob):
    return pickle.loads(blob)      # written by our own training job into our own database
