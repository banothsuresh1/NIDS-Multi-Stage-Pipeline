"""
Stage 12 — Streaming Evaluation.

Simulates online deployment by replaying the test-session stream through
sliding time windows, measuring detection quality (macro-F1), per-window
inference latency, throughput, and alert rate — the metrics a SOC would
track for a real-time NIDS.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config import RISK_SUSPICIOUS_THRESH, STREAM_STRIDE, STREAM_WINDOW_SIZE
from src.nids.risk import build_meta_features


def make_synthetic_timestamps(n_sessions: int, spacing_seconds: float = 2.0) -> np.ndarray:
    return np.linspace(0, n_sessions * spacing_seconds, n_sessions)


def evaluate_streaming(
    session_ids, session_timestamps: np.ndarray, y_true: np.ndarray,
    R_test: np.ndarray, SP_t_test: np.ndarray, TC_t_test: np.ndarray,
    G_w_test: np.ndarray, dt_inv_test: np.ndarray,
    meta_lr, window_size: int = STREAM_WINDOW_SIZE, stride: int = STREAM_STRIDE,
) -> pd.DataFrame:
    """Replay the test stream through sliding windows and record per-window metrics."""
    session_timestamps = np.asarray(session_timestamps)
    y_true = np.asarray(y_true)
    duration = float(session_timestamps.max()) if len(session_timestamps) else 0.0

    if duration >= window_size:
        n_windows = int(np.floor((duration - window_size) / stride)) + 1
    else:
        n_windows = 1
    window_starts = [i * stride for i in range(n_windows)]

    rows = []
    for window_start in window_starts:
        window_end = window_start + window_size
        mask = (session_timestamps >= window_start) & (session_timestamps < window_end)
        n_sessions = int(mask.sum())

        t_start = time.perf_counter()
        if n_sessions > 0:
            X_meta_window = build_meta_features(
                R_test[mask], SP_t_test[mask], TC_t_test[mask], G_w_test[mask], dt_inv_test[mask]
            )
            risk_scores = meta_lr.predict_proba(X_meta_window)
            attack_idx = list(meta_lr.classes_).index(1) if 1 in meta_lr.classes_ else -1
            risk_scores = risk_scores[:, attack_idx]
        else:
            risk_scores = np.array([])
        t_end = time.perf_counter()

        latency_ms = max((t_end - t_start) * 1000.0, 1e-6)
        throughput = n_sessions / max(latency_ms / 1000.0, 1e-9)

        if n_sessions > 0:
            y_pred_binary = (risk_scores >= RISK_SUSPICIOUS_THRESH).astype(int)
            y_true_binary = (y_true[mask] != 0).astype(int)
            macro_f1 = f1_score(
                y_true_binary, y_pred_binary, average="macro", labels=[0, 1], zero_division=0,
            )
            alert_rate = float(y_pred_binary.mean())
        else:
            macro_f1 = 0.0
            alert_rate = 0.0

        rows.append({
            "window_start": window_start,
            "window_end": window_end,
            "n_sessions": n_sessions,
            "macro_f1": macro_f1,
            "latency_ms": latency_ms,
            "throughput": throughput,
            "alert_rate": alert_rate,
        })

    return pd.DataFrame(rows)
