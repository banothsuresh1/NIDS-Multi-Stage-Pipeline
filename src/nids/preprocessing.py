"""
Stage 2 — Preprocessing & Imbalance Handling.

Handles CIC-IDS2017 numeric pathologies (inf values from flow-rate ratios,
extreme outliers) with a train-fit / val-test-transform contract to avoid
temporal leakage, and provides a two-branch imbalance strategy:
  Branch A (raw, class-weighted)      -> used by BiLSTM / sequence models
  Branch B (SMOTE+ENN augmented)      -> used by RF / XGBoost tabular models
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from config import (
    ENN_N_NEIGHBORS,
    OUTLIER_CLIP_PERCENTILE,
    SMOTE_K_NEIGHBORS,
    SMOTE_MIN_SAMPLES,
)

NON_FEATURE_COLS = {"Label", "Day", "SessionID", "Token", "TokenID", "LabelID"}

# 5-tuple / timestamp identifier columns are excluded from the ML feature
# set (standard CIC-IDS2017 practice — training on raw lab IPs/ports
# overfits to the capture topology). Session reconstruction (Stage 4) and
# event-token assignment (Stage 5) run AFTER this preprocessing stage and
# need these columns UNSCALED, so they must never be MinMax-transformed.
IDENTIFIER_COL_VARIANTS = [
    "Src IP", " Source IP", "Source IP", "Src_IP",
    "Dst IP", " Destination IP", "Destination IP", "Dst_IP",
    "Src Port", " Source Port", "Source Port",
    "Dst Port", " Destination Port", "Destination Port",
    "Protocol", " Protocol",
    "Timestamp", " Timestamp", "timestamp",
    "Flow ID", " Flow ID",
]


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return all ML feature columns: numeric flow-statistic columns, excluding
    metadata/label columns AND raw 5-tuple/timestamp identifier columns
    (those are needed unscaled by session reconstruction and event encoding).
    """
    exclude = NON_FEATURE_COLS | set(IDENTIFIER_COL_VARIANTS)
    return [c for c in df.columns if c not in exclude]


def preprocess(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler, List[str]]:
    """
    Clean inf/NaN, clip outliers (fit on train), and MinMax-scale all splits
    using statistics fit exclusively on the training split.
    """
    feat_cols = get_feature_cols(df_train)
    # Keep only columns that are actually numeric.
    numeric_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df_train[c])]

    train = df_train.copy()
    val = df_val.copy()
    test = df_test.copy()

    for df in (train, val, test):
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    train = train.dropna(subset=numeric_cols).reset_index(drop=True)
    val = val.dropna(subset=numeric_cols).reset_index(drop=True)
    test = test.dropna(subset=numeric_cols).reset_index(drop=True)

    # Clip outliers at the OUTLIER_CLIP_PERCENTILE, fit on train only.
    upper_bounds = train[numeric_cols].quantile(OUTLIER_CLIP_PERCENTILE / 100.0)
    lower_bounds = train[numeric_cols].quantile(1 - OUTLIER_CLIP_PERCENTILE / 100.0)
    for df in (train, val, test):
        df[numeric_cols] = df[numeric_cols].clip(lower=lower_bounds, upper=upper_bounds, axis=1)

    scaler = MinMaxScaler()
    train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
    val[numeric_cols] = scaler.transform(val[numeric_cols])
    test[numeric_cols] = scaler.transform(test[numeric_cols])

    return train, val, test, scaler, numeric_cols


def encode_labels(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> Tuple[LabelEncoder, np.ndarray, int]:
    """
    Fit LabelEncoder on training labels only. Unseen labels in val/test
    (rare in CIC-IDS2017 but possible across the chronological split) are
    mapped to the majority class "BENIGN" to keep downstream arrays valid.
    """
    le = LabelEncoder()
    le.fit(df_train["Label"].astype(str))
    classes = le.classes_
    known = set(classes)
    fallback = "BENIGN" if "BENIGN" in known else classes[0]

    for df in (df_train, df_val, df_test):
        df["Label"] = df["Label"].astype(str)
        df.loc[~df["Label"].isin(known), "Label"] = fallback
        df["LabelID"] = le.transform(df["Label"])

    return le, classes, len(classes)


def compute_class_weights(df_train: pd.DataFrame, le: LabelEncoder, K: int) -> Dict[int, float]:
    """Inverse-frequency class weights: W_c = N_train / (K * N_c)."""
    n_train = len(df_train)
    counts = df_train["LabelID"].value_counts().to_dict()
    weights = {}
    for c in range(K):
        n_c = counts.get(c, 0)
        weights[c] = n_train / (K * n_c) if n_c > 0 else 0.0
    return weights


def smote_knn_augment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_counts: Dict[int, int],
    K: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment minority classes (count < SMOTE_MIN_SAMPLES) with SMOTE, then
    clean borderline/noisy points with Edited Nearest Neighbours. Falls back
    to the original arrays if SMOTE cannot run (e.g. too few neighbours).
    """
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import EditedNearestNeighbours

        minority_classes = [c for c, n in class_counts.items() if 0 < n < SMOTE_MIN_SAMPLES]
        if not minority_classes:
            return X_train, y_train

        # k_neighbors must be < smallest class size for SMOTE to run.
        smallest = min(class_counts.get(c, 1) for c in minority_classes)
        k_neighbors = max(1, min(SMOTE_K_NEIGHBORS, smallest - 1))
        if k_neighbors < 1:
            return X_train, y_train

        sampling_strategy = {c: SMOTE_MIN_SAMPLES for c in minority_classes}
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=seed,
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)

        enn_neighbors = min(ENN_N_NEIGHBORS, len(X_res) - 1)
        if enn_neighbors >= 1:
            enn = EditedNearestNeighbours(n_neighbors=enn_neighbors)
            X_res, y_res = enn.fit_resample(X_res, y_res)

        return X_res, y_res
    except Exception as exc:  # pragma: no cover - defensive fallback path
        print(f"[smote_knn_augment] SMOTE/ENN failed ({exc}); returning original data.")
        return X_train, y_train
