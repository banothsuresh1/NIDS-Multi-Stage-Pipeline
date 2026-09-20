import numpy as np
import pandas as pd

from config import FUSION_GRID
from src.nids.fusion import calibrate_fusion_weights, flow_probs_to_session, fuse


def _rand_probs(n, k, seed):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, k))
    return raw / raw.sum(axis=1, keepdims=True)


def test_flow_probs_to_session_shape():
    df_sess = pd.DataFrame({"SessionID": ["s1", "s1", "s2", "s2", "s2"]})
    P_flow = _rand_probs(5, 4, seed=1)
    session_ids = ["s1", "s2"]
    out = flow_probs_to_session(df_sess, P_flow, session_ids)
    assert out.shape == (2, 4)


def test_fuse_probabilities_sum_to_one():
    K = 5
    n = 20
    P_A, P_B, P_C = _rand_probs(n, K, 1), _rand_probs(n, K, 2), _rand_probs(n, K, 3)
    SP_t = np.random.default_rng(4).random(n)
    TC_t = np.random.default_rng(5).random(n)
    R = fuse(P_A, P_B, P_C, SP_t, TC_t, 0.35, 0.25, 0.25, 0.10, 0.05)
    sums = R.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_calibrate_fusion_weights_returns_valid_tuple():
    K = 5
    n = 40
    P_A, P_B, P_C = _rand_probs(n, K, 1), _rand_probs(n, K, 2), _rand_probs(n, K, 3)
    SP_t = np.random.default_rng(4).random(n)
    TC_t = np.random.default_rng(5).random(n)
    y_val = np.random.default_rng(6).integers(0, K, n)

    small_grid = {
        "w_A": [0.3, 0.5], "w_B": [0.2], "w_C": [0.2], "w_S": [0.1], "w_T": [0.05],
    }
    weights = calibrate_fusion_weights(P_A, P_B, P_C, SP_t, TC_t, y_val, small_grid)
    assert len(weights) == 5
    assert all(w > 0 for w in weights)


def test_fuse_equal_weights_equals_average():
    K = 4
    n = 10
    P_A, P_B, P_C = _rand_probs(n, K, 1), _rand_probs(n, K, 2), _rand_probs(n, K, 3)
    SP_t = np.zeros(n)
    TC_t = np.zeros(n)
    R = fuse(P_A, P_B, P_C, SP_t, TC_t, 1.0, 1.0, 1.0, 0.0, 0.0)
    expected = (P_A + P_B + P_C) / 3.0
    assert np.allclose(R, expected, atol=1e-6)
