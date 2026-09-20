import numpy as np

from config import K as K_DEFAULT
from src.nids.risk import build_meta_features, predict_risk, risk_to_tier, train_meta_learner


def _fit_meta_learner(n=200, k=5, seed=0):
    rng = np.random.default_rng(seed)
    R = rng.random((n, k))
    R = R / R.sum(axis=1, keepdims=True)
    SP_t, TC_t, G_w, dt_inv = (rng.random(n) for _ in range(4))
    X_meta = build_meta_features(R, SP_t, TC_t, G_w, dt_inv)
    y_risk = rng.integers(0, 2, n)
    # ensure both classes present
    y_risk[0], y_risk[1] = 0, 1
    meta_lr = train_meta_learner(X_meta, y_risk)
    return meta_lr, X_meta, k


def test_predict_risk_in_range():
    meta_lr, X_meta, _ = _fit_meta_learner()
    risk = predict_risk(meta_lr, X_meta)
    assert (risk >= 0.0).all() and (risk <= 1.0).all()


def test_risk_to_tier_benign():
    assert risk_to_tier(0.10) == "BENIGN"


def test_risk_to_tier_suspicious():
    assert risk_to_tier(0.50) == "SUSPICIOUS"


def test_risk_to_tier_attack():
    assert risk_to_tier(0.80) == "ATTACK"


def test_build_meta_features_shape():
    n, k = 30, 5
    rng = np.random.default_rng(1)
    R = rng.random((n, k))
    SP_t, TC_t, G_w, dt_inv = (rng.random(n) for _ in range(4))
    X_meta = build_meta_features(R, SP_t, TC_t, G_w, dt_inv)
    assert X_meta.shape == (n, k + 4)
