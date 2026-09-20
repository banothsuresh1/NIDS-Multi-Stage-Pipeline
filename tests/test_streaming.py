import numpy as np

from src.nids.risk import build_meta_features, train_meta_learner
from src.nids.streaming import evaluate_streaming, make_synthetic_timestamps


def _stream_inputs(n=100, k=5, seed=0, window_size=60, stride=10, spacing=2.0):
    rng = np.random.default_rng(seed)
    R = rng.random((n, k))
    R = R / R.sum(axis=1, keepdims=True)
    SP_t, TC_t, G_w, dt_inv = (rng.random(n) for _ in range(4))
    y_true = rng.integers(0, k, n)

    X_meta = build_meta_features(R, SP_t, TC_t, G_w, dt_inv)
    y_risk = (y_true != 0).astype(int)
    y_risk[0], y_risk[1] = 0, 1
    meta_lr = train_meta_learner(X_meta, y_risk)

    timestamps = make_synthetic_timestamps(n, spacing_seconds=spacing)
    session_ids = [f"s{i}" for i in range(n)]
    return session_ids, timestamps, y_true, R, SP_t, TC_t, G_w, dt_inv, meta_lr


def test_evaluate_streaming_columns():
    session_ids, ts, y_true, R, SP_t, TC_t, G_w, dt_inv, meta_lr = _stream_inputs()
    results_df = evaluate_streaming(
        session_ids, ts, y_true, R, SP_t, TC_t, G_w, dt_inv, meta_lr,
        window_size=60, stride=10,
    )
    for col in ("window_start", "n_sessions", "macro_f1", "latency_ms"):
        assert col in results_df.columns


def test_evaluate_streaming_window_count():
    n, spacing, window_size, stride = 100, 2.0, 60, 10
    session_ids, ts, y_true, R, SP_t, TC_t, G_w, dt_inv, meta_lr = _stream_inputs(
        n=n, spacing=spacing, window_size=window_size, stride=stride
    )
    duration = ts.max()
    expected_windows = int(np.floor((duration - window_size) / stride)) + 1

    results_df = evaluate_streaming(
        session_ids, ts, y_true, R, SP_t, TC_t, G_w, dt_inv, meta_lr,
        window_size=window_size, stride=stride,
    )
    assert len(results_df) == expected_windows


def test_evaluate_streaming_latency_positive():
    session_ids, ts, y_true, R, SP_t, TC_t, G_w, dt_inv, meta_lr = _stream_inputs()
    results_df = evaluate_streaming(
        session_ids, ts, y_true, R, SP_t, TC_t, G_w, dt_inv, meta_lr,
        window_size=60, stride=10,
    )
    assert (results_df["latency_ms"] > 0).all()


def test_evaluate_streaming_macro_f1_range():
    session_ids, ts, y_true, R, SP_t, TC_t, G_w, dt_inv, meta_lr = _stream_inputs()
    results_df = evaluate_streaming(
        session_ids, ts, y_true, R, SP_t, TC_t, G_w, dt_inv, meta_lr,
        window_size=60, stride=10,
    )
    assert ((results_df["macro_f1"] >= 0.0) & (results_df["macro_f1"] <= 1.0)).all()
