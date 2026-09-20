"""
Stage 3 — Feature Engineering.

Splits the CIC-IDS2017 feature space into four semantically grounded groups
using mutual information (MI) ranking plus keyword-based domain rules, then
assembles the per-model feature sets used downstream:
  RF   uses GROUP_A (top MI, general)    U GROUP_B (handshake/flag features)
  XGB  uses GROUP_B (handshake/flag)     U GROUP_D (behavioral flag counts)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from config import GROUP_A_SIZE, GROUP_B_SIZE, GROUP_C_SIZE, GROUP_D_SIZE

GROUP_B_KEYWORDS = [
    "Port", "Protocol", "PSH", "URG", "FIN", "SYN", "RST", "ACK", "ECE",
    "CWE", "Header", "Init_Win",
]
GROUP_C_KEYWORDS = ["Duration", "IAT"]
GROUP_D_KEYWORDS = ["Flag Cnt", "flag count", "Flags"]


def compute_mi_scores(
    X_train: pd.DataFrame, y_train: np.ndarray, feat_cols: List[str], seed: int
) -> pd.Series:
    """Mutual information between each feature and the class label."""
    X = X_train[feat_cols] if isinstance(X_train, pd.DataFrame) else X_train
    mi = mutual_info_classif(X, y_train, random_state=seed)
    return pd.Series(mi, index=feat_cols).sort_values(ascending=False)


def _match_keywords(feat_cols: List[str], keywords: List[str]) -> List[str]:
    matched = []
    for col in feat_cols:
        col_lower = col.lower()
        if any(kw.lower() in col_lower for kw in keywords):
            matched.append(col)
    return matched


def _pad_group(group: List[str], target_size: int, mi_ranked: List[str], used: set) -> List[str]:
    """Pad a group up to target_size using the next-highest MI features not yet used."""
    group = list(dict.fromkeys(group))  # de-dup, preserve order
    if len(group) >= target_size:
        return group[:target_size]
    for feat in mi_ranked:
        if len(group) >= target_size:
            break
        if feat not in group and feat not in used:
            group.append(feat)
    return group


def assign_feature_groups(
    feat_cols: List[str], mi_scores: pd.Series
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    GROUP_B: TCP handshake / flag / header features (keyword match)
    GROUP_C: duration / inter-arrival-time features (keyword match)
    GROUP_D: aggregate flag-count / behavioral features (keyword match)
    GROUP_A: remaining top-MI general features (size = GROUP_A_SIZE)
    Each group is padded to its target size using MI ranking if undersized.
    """
    mi_ranked = list(mi_scores.index)

    # Keyword groups are assigned with strict priority D > B > C: D's
    # "Flag Cnt/flag count/Flags" keywords are more specific to behavioral
    # flag-count aggregates than B's individual-letter keywords (SYN, PSH,
    # ...), which in raw CIC-IDS2017 column names would otherwise overlap
    # completely (e.g. "SYN Flag Count" matches both). Remaining top-MI
    # features form GROUP_A.
    group_d = _match_keywords(feat_cols, GROUP_D_KEYWORDS)
    group_b = [f for f in _match_keywords(feat_cols, GROUP_B_KEYWORDS) if f not in group_d]
    group_c = [
        f for f in _match_keywords(feat_cols, GROUP_C_KEYWORDS)
        if f not in group_d and f not in group_b
    ]

    used = set(group_b) | set(group_c) | set(group_d)
    group_a = [f for f in mi_ranked if f not in used][:GROUP_A_SIZE]
    used |= set(group_a)

    group_b = _pad_group(group_b, min(GROUP_B_SIZE, len(feat_cols)), mi_ranked, used)
    used |= set(group_b)
    group_c = _pad_group(group_c, min(GROUP_C_SIZE, len(feat_cols)), mi_ranked, used)
    used |= set(group_c)
    group_d = _pad_group(group_d, min(GROUP_D_SIZE, len(feat_cols)), mi_ranked, used)
    used |= set(group_d)

    group_a = _pad_group(group_a, min(GROUP_A_SIZE, len(feat_cols)), mi_ranked, used)

    return group_a, group_b, group_c, group_d


def get_model_features(
    GROUP_A: List[str], GROUP_B: List[str], GROUP_C: List[str], GROUP_D: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """RF -> A U B, LSTM -> token sequence (no tabular features), XGB -> B U D."""
    rf_features = list(dict.fromkeys(GROUP_A + GROUP_B))
    lstm_features: List[str] = []  # BiLSTM consumes the token sequence, not tabular features
    xgb_features = list(dict.fromkeys(GROUP_B + GROUP_D))
    return rf_features, lstm_features, xgb_features
