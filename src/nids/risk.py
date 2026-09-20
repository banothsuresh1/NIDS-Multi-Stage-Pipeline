"""
Stage 11 — Adaptive Risk Meta-Learner.

Combines the fused per-class evidence vector (R_t) with the three
structural signals (SP_t pattern-match, TC_t graph-transition-consistency,
G_w graph-centrality) and an inverse-dwell-time proxy (dt_inv) into a
(K+4)-dimensional meta-feature vector, then trains a lightweight logistic
regression meta-learner to output a single calibrated Risk_t in [0, 1].
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression

from config import META_LR_C, RISK_BENIGN_THRESH, RISK_SUSPICIOUS_THRESH, SEED


def build_meta_features(R: np.ndarray, SP_t: np.ndarray, TC_t: np.ndarray,
                         G_w: np.ndarray, dt_inv: np.ndarray) -> np.ndarray:
    """Concatenate [R (K-dim), SP_t, TC_t, G_w, dt_inv] -> (n_sessions, K+4)."""
    SP_t = np.asarray(SP_t).reshape(-1, 1)
    TC_t = np.asarray(TC_t).reshape(-1, 1)
    G_w = np.asarray(G_w).reshape(-1, 1)
    dt_inv = np.asarray(dt_inv).reshape(-1, 1)
    return np.hstack([R, SP_t, TC_t, G_w, dt_inv])


def compute_dt_inv(sessions_dict: Dict[str, dict]) -> Dict[str, float]:
    """Proxy dwell-time-inverse: 1 / max(n_tokens - 1, 1)."""
    return {
        sid: 1.0 / max(len(s["tokens"]) - 1, 1)
        for sid, s in sessions_dict.items()
    }


def train_meta_learner(X_meta_val: np.ndarray, y_risk_val: np.ndarray) -> LogisticRegression:
    """Binary meta-learner: BENIGN=0, any attack=1."""
    meta_lr = LogisticRegression(C=META_LR_C, max_iter=1000, random_state=SEED)
    meta_lr.fit(X_meta_val, y_risk_val)
    return meta_lr


def predict_risk(meta_lr: LogisticRegression, X_meta: np.ndarray) -> np.ndarray:
    proba = meta_lr.predict_proba(X_meta)
    attack_idx = list(meta_lr.classes_).index(1) if 1 in meta_lr.classes_ else -1
    return proba[:, attack_idx]


def risk_to_tier(risk_score: float, benign_thresh: float = RISK_BENIGN_THRESH,
                  suspicious_thresh: float = RISK_SUSPICIOUS_THRESH) -> str:
    if risk_score < benign_thresh:
        return "BENIGN"
    if risk_score >= suspicious_thresh:
        return "ATTACK"
    return "SUSPICIOUS"


def classify_sessions(Risk_t_array: np.ndarray) -> List[str]:
    return [risk_to_tier(r) for r in Risk_t_array]
