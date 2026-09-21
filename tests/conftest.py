"""
Shared pytest fixtures: a synthetic mini-dataset that mimics CIC-IDS2017's
structure (leading-space columns, duplicate header rows, inf values, a
5-class imbalanced label distribution) so the full pipeline can be exercised
end-to-end without the real ~50 GB dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DAY_FILES, SEED, VOCAB
from src.nids.events import encode_sessions
from src.nids.preprocessing import (
    compute_class_weights,
    encode_labels,
    get_feature_cols,
    preprocess,
    smote_knn_augment,
)
from src.nids.sessions import reconstruct_sessions

N_ROWS = 1000
CLASSES = ["BENIGN", "DoS Hulk", "PortScan", "FTP-Patator", "Bot"]
CLASS_PROBS = [0.55, 0.20, 0.15, 0.08, 0.02]  # Bot is intentionally rare (< SMOTE_MIN_SAMPLES)


def _make_raw_frame(n_rows: int, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    labels = rng.choice(CLASSES, size=n_rows, p=CLASS_PROBS)
    is_attack = (labels != "BENIGN").astype(float)
    # A distinct per-class fingerprint (on top of the attack/benign shift)
    # keeps classes separable enough that SMOTE+ENN cleans rather than
    # erases the oversampled minority class, and that RF/XGB/BiLSTM reach
    # non-trivial accuracy on this synthetic mini-dataset.
    class_idx = np.array([CLASSES.index(l) for l in labels], dtype=float)

    def noisy(base, scale, shift_for_attack=0.0, shift_per_class=0.0):
        return (
            base
            + shift_for_attack * is_attack
            + shift_per_class * class_idx
            + rng.normal(0, scale, n_rows)
        )

    src_ip = rng.integers(1, 50, n_rows)
    dst_ip = rng.integers(1, 50, n_rows)
    dst_port = np.where(
        rng.random(n_rows) < 0.3,
        rng.choice([22, 21, 23, 3389, 445], n_rows),
        rng.integers(1024, 65535, n_rows),
    )

    data = {
        " Source IP": [f"10.0.0.{i}" for i in src_ip],
        " Destination IP": [f"10.0.1.{i}" for i in dst_ip],
        " Source Port": rng.integers(1024, 65535, n_rows),
        " Destination Port": dst_port,
        " Protocol": rng.choice([6, 17], n_rows),
        "Timestamp": pd.date_range("2017-07-03 09:00:00", periods=n_rows, freq="2s"),
        " Flow Duration": np.abs(noisy(500_000, 150_000, shift_for_attack=-100_000, shift_per_class=40_000)),
        " Total Fwd Packets": np.abs(noisy(10, 3, shift_for_attack=50, shift_per_class=8)).round(),
        " Total Backward Packets": np.abs(noisy(8, 3, shift_for_attack=2, shift_per_class=3)).round(),
        "Total Length of Fwd Packets": np.abs(noisy(2000, 500, shift_for_attack=3000, shift_per_class=1500)),
        " Total Length of Bwd Packets": np.abs(noisy(1500, 500, shift_for_attack=200, shift_per_class=1200)),
        " Fwd Packet Length Max": np.abs(noisy(500, 150)),
        " Fwd Packet Length Min": np.abs(noisy(20, 10)),
        " Fwd Packet Length Mean": np.abs(noisy(150, 60)),
        " Fwd Packet Length Std": np.abs(noisy(80, 30)),
        "Bwd Packet Length Max": np.abs(noisy(450, 140)),
        " Bwd Packet Length Min": np.abs(noisy(15, 8)),
        " Bwd Packet Length Mean": np.abs(noisy(130, 55)),
        " Bwd Packet Length Std": np.abs(noisy(70, 25)),
        "Flow Bytes/s": np.abs(noisy(5000, 1500, shift_for_attack=8000, shift_per_class=6000)),
        " Flow Packets/s": np.abs(noisy(20, 8, shift_for_attack=15000, shift_per_class=3000)),
        " Flow IAT Mean": np.abs(noisy(4000, 1500)),
        " Flow IAT Std": np.abs(noisy(2000, 900)),
        " Flow IAT Max": np.abs(noisy(9000, 3000)),
        " Flow IAT Min": np.abs(noisy(100, 50)),
        "Fwd IAT Total": np.abs(noisy(400_000, 150_000)),
        " Fwd IAT Mean": np.abs(noisy(3500, 1200)),
        " Fwd IAT Std": np.abs(noisy(1800, 700)),
        " Fwd IAT Max": np.abs(noisy(8500, 2800)),
        " Fwd IAT Min": np.abs(noisy(90, 40)),
        "Bwd IAT Total": np.abs(noisy(380_000, 140_000)),
        " Bwd IAT Mean": np.abs(noisy(3300, 1100)),
        " Bwd IAT Std": np.abs(noisy(1700, 650)),
        " Bwd IAT Max": np.abs(noisy(8200, 2700)),
        " Bwd IAT Min": np.abs(noisy(85, 35)),
        "Fwd PSH Flags": rng.integers(0, 2, n_rows),
        " Fwd URG Flags": rng.integers(0, 2, n_rows),
        " SYN Flag Count": rng.integers(0, 2, n_rows),
        " ACK Flag Count": rng.integers(0, 2, n_rows),
        " FIN Flag Count": rng.integers(0, 2, n_rows),
        " RST Flag Count": rng.integers(0, 2, n_rows),
        " PSH Flag Count": rng.integers(0, 2, n_rows),
        " URG Flag Count": rng.integers(0, 2, n_rows),
        " ECE Flag Count": rng.integers(0, 2, n_rows),
        "CWE Flag Count": rng.integers(0, 2, n_rows),
        "Init_Win_bytes_forward": rng.integers(-1, 65535, n_rows),
        " Init_Win_bytes_backward": rng.integers(-1, 65535, n_rows),
        " Active Mean": np.abs(noisy(1000, 400)),
        " Idle Mean": np.abs(noisy(2000, 800)),
        " Label": labels,
    }
    df = pd.DataFrame(data)

    # Inject a handful of inf values, mimicking CIC-IDS2017's Flow Bytes/s pathology.
    inf_idx = rng.choice(n_rows, size=3, replace=False)
    df.loc[inf_idx, "Flow Bytes/s"] = np.inf

    return df


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Raw synthetic frame with leading-space columns, mimicking a CIC-IDS2017 CSV read."""
    df = _make_raw_frame(N_ROWS, seed=SEED)
    # Simulate a duplicate embedded header row (a known CIC-IDS2017 CSV artifact).
    dup_row = {col: col.strip() for col in df.columns}
    dup_row[" Label"] = "Label"
    df = pd.concat([df, pd.DataFrame([dup_row])], ignore_index=True)
    return df


@pytest.fixture(scope="session")
def synthetic_csv_dir(tmp_path_factory) -> Path:
    """Writes a synthetic Monday CSV to disk under the DAY_FILES[1] naming convention."""
    raw_df = _make_raw_frame(300, seed=SEED)
    dup_row = {col: col.strip() for col in raw_df.columns}
    dup_row[" Label"] = "Label"
    raw_df = pd.concat([raw_df, pd.DataFrame([dup_row])], ignore_index=True)

    out_dir = tmp_path_factory.mktemp("cic_ids2017")
    raw_df.to_csv(out_dir / DAY_FILES[1], index=False)
    raw_df2 = _make_raw_frame(200, seed=SEED + 1)
    raw_df2.to_csv(out_dir / DAY_FILES[2], index=False)
    return out_dir


@pytest.fixture(scope="session")
def synthetic_csv_dir_all_days(tmp_path_factory) -> Path:
    """Writes all seven synthetic day CSVs, for testing pooled (not per-day) loading."""
    out_dir = tmp_path_factory.mktemp("cic_ids2017_all_days")
    for day in range(1, 8):
        df = _make_raw_frame(60, seed=SEED + day)
        df.to_csv(out_dir / DAY_FILES[day], index=False)
    return out_dir


def _clean(df_raw: pd.DataFrame, day: int) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    df = df[df["Label"] != "Label"].reset_index(drop=True)
    df["Day"] = day
    return df


@pytest.fixture(scope="session")
def synthetic_split():
    """Chronologically-flavoured train/val/test split of already-cleaned synthetic data."""
    df_train = _clean(_make_raw_frame(500, seed=SEED), day=1)
    df_val = _clean(_make_raw_frame(250, seed=SEED + 1), day=3)
    df_test = _clean(_make_raw_frame(250, seed=SEED + 2), day=4)
    return df_train, df_val, df_test


@pytest.fixture(scope="session")
def preprocessed_data(synthetic_split):
    df_train, df_val, df_test = synthetic_split
    train, val, test, scaler, feat_cols = preprocess(df_train, df_val, df_test)
    le, classes, K = encode_labels(train, val, test)
    class_weights = compute_class_weights(train, le, K)

    X_train_A = train[feat_cols].values
    y_train_A = train["LabelID"].values
    X_val = val[feat_cols].values
    y_val = val["LabelID"].values
    X_test = test[feat_cols].values
    y_test = test["LabelID"].values

    class_counts = train["LabelID"].value_counts().to_dict()
    X_aug, y_aug = smote_knn_augment(X_train_A, y_train_A, class_counts, K, SEED)

    return {
        "X_train_A": X_train_A, "y_train_A": y_train_A,
        "X_aug": X_aug, "y_aug": y_aug,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler, "feat_cols": feat_cols,
        "le": le, "classes": classes, "K": K,
        "class_weights": class_weights,
        "train": train, "val": val, "test": test,
    }


@pytest.fixture(scope="session")
def synthetic_sessions(synthetic_split):
    df_train, df_val, df_test = synthetic_split

    df_train_sess = reconstruct_sessions(df_train)
    df_val_sess = reconstruct_sessions(df_val)
    df_test_sess = reconstruct_sessions(df_test)

    _, sessions_train = encode_sessions(df_train_sess)
    _, sessions_val = encode_sessions(df_val_sess)
    _, sessions_test = encode_sessions(df_test_sess)

    return sessions_train, sessions_val, sessions_test


@pytest.fixture(scope="session")
def trained_rf(preprocessed_data):
    from src.nids.models import train_rf

    d = preprocessed_data
    return train_rf(d["X_aug"], d["y_aug"], d["class_weights"], SEED)


@pytest.fixture(scope="session")
def trained_xgb(preprocessed_data):
    from src.nids.models import train_xgboost

    d = preprocessed_data
    return train_xgboost(
        d["X_aug"], d["y_aug"], d["class_weights"], d["X_val"], d["y_val"], d["K"], SEED
    )


@pytest.fixture(scope="session")
def attack_graph_fixture(synthetic_sessions):
    from src.nids.attack_graph import build_attack_graph

    sessions_train, _, _ = synthetic_sessions
    return build_attack_graph(sessions_train)
