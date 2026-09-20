"""
Stage 7 — Adaptive Evidence Fusion.

Pools flow-level classifier probabilities to session level and fuses the
three classifier views (RF, BiLSTM, XGBoost) with the pattern-mining
consistency score (SP_t, Stage 8) and the graph temporal-consistency score
(TC_t, Stages 9-10) into a single per-session, per-class evidence vector.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score

from config import K as K_DEFAULT


def flow_probs_to_session(df_sess, P_flow: np.ndarray, session_ids: Sequence[str]) -> np.ndarray:
    """Mean-pool flow-level probabilities within each session (order = session_ids)."""
    n_classes = P_flow.shape[1]
    df_reset = df_sess.reset_index(drop=True)
    session_col = df_reset["SessionID"].values

    # Map each flow row -> its position in P_flow (assumes row-aligned order).
    groups: Dict[str, List[int]] = {}
    for idx, sid in enumerate(session_col):
        if idx >= len(P_flow):
            break
        groups.setdefault(sid, []).append(idx)

    out = np.full((len(session_ids), n_classes), 1.0 / n_classes, dtype=np.float64)
    for i, sid in enumerate(session_ids):
        idxs = groups.get(sid)
        if idxs:
            out[i] = P_flow[idxs].mean(axis=0)
    return out


def fuse(P_A: np.ndarray, P_B: np.ndarray, P_C: np.ndarray,
         SP_t: np.ndarray, TC_t: np.ndarray,
         w_A: float, w_B: float, w_C: float, w_S: float, w_T: float) -> np.ndarray:
    """
    R_t = normalized weighted classifier blend + uniformly distributed
    scalar consistency signals (SP_t, TC_t) spread across all K classes.
    """
    K = P_A.shape[1]
    classifier_weight_sum = w_A + w_B + w_C
    if classifier_weight_sum <= 0:
        classifier_weight_sum = 1.0

    R = (w_A * P_A + w_B * P_B + w_C * P_C) / classifier_weight_sum

    SP_t = np.asarray(SP_t).reshape(-1, 1)
    TC_t = np.asarray(TC_t).reshape(-1, 1)

    R = R + w_S * (SP_t / K) + w_T * (TC_t / K)

    row_sums = R.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    R = R / row_sums
    return R


def calibrate_fusion_weights(
    P_A_val: np.ndarray, P_B_val: np.ndarray, P_C_val: np.ndarray,
    SP_t_val: np.ndarray, TC_t_val: np.ndarray,
    y_val: np.ndarray, fusion_grid: Dict[str, List[float]],
) -> Tuple[float, float, float, float, float]:
    """Grid search over FUSION_GRID, maximizing macro-F1 on the validation set."""
    best_f1 = -1.0
    best_weights = (0.35, 0.25, 0.25, 0.10, 0.05)

    for w_A, w_B, w_C, w_S, w_T in product(
        fusion_grid["w_A"], fusion_grid["w_B"], fusion_grid["w_C"],
        fusion_grid["w_S"], fusion_grid["w_T"],
    ):
        R = fuse(P_A_val, P_B_val, P_C_val, SP_t_val, TC_t_val, w_A, w_B, w_C, w_S, w_T)
        y_pred = np.argmax(R, axis=1)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = (w_A, w_B, w_C, w_S, w_T)

    print(f"[calibrate_fusion_weights] Best weights: {best_weights} | val macro-F1: {best_f1:.4f}")
    return best_weights
