"""
Stage 5 — Behavioral Event Encoding.

Converts each flow row into one of 10 coarse behavioral tokens (VOCAB) that
summarise handshake state, scanning, auth outcome and data-transfer volume,
then assembles per-session token sequences for the BiLSTM branch.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import AUTH_PORTS, MAX_SEQ_LEN, TOKEN2ID

# Column-name variants encountered across CIC-IDS2017 day files.
COL_VARIANTS: Dict[str, List[str]] = {
    "dst_port": ["Dst Port", " Destination Port", "Destination Port"],
    "duration": ["Flow Duration", " Flow Duration"],
    "total_fwd_pkts": ["Tot Fwd Pkts", "Total Fwd Packets", " Total Fwd Packets"],
    "total_bwd_pkts": ["Tot Bwd Pkts", "Total Backward Packets", " Total Backward Packets"],
    "total_fwd_bytes": ["TotLen Fwd Pkts", "Total Length of Fwd Packets", " Total Length of Fwd Packets"],
    "total_bwd_bytes": ["TotLen Bwd Pkts", "Total Length of Bwd Packets", " Total Length of Bwd Packets"],
    "syn": ["SYN Flag Cnt", "SYN Flag Count", " SYN Flag Count"],
    "ack": ["ACK Flag Cnt", "ACK Flag Count", " ACK Flag Count"],
    "fin": ["FIN Flag Cnt", "FIN Flag Count", " FIN Flag Count"],
    "rst": ["RST Flag Cnt", "RST Flag Count", " RST Flag Count"],
}

# assign_token()'s rules are thresholds in raw units (packets, bytes,
# microseconds -- e.g. "pkt_rate > 10000", "total_bytes < 100"). If the
# dataframe it reads from has already been MinMax-scaled to [0, 1] (as
# happens when Stage 5 runs on Stage 2's preprocessed output, not the raw
# flow data), every one of those thresholds breaks silently -- duration
# divided by 1e6 collapses to ~0, so pkt_rate explodes and DOS_INDICATOR
# fires almost unconditionally regardless of true traffic behavior.
# preprocess() (preprocessing.py) snapshots these columns under a
# "__raw__" prefix before scaling; get_col() below prefers that snapshot
# when present and falls back to the plain column (for callers that pass
# an already-raw/unscaled dataframe directly, e.g. tests).
RAW_COL_PREFIX = "__raw__"
RAW_MAGNITUDE_COLS: List[str] = sorted({
    name
    for key, variants in COL_VARIANTS.items()
    if key != "dst_port"  # an identifier column, never scaled in the first place
    for name in variants
})


def get_col(row: pd.Series, names: List[str], default: float = 0.0) -> float:
    """
    Try each column-name variant in order, preferring the unscaled
    "__raw__"-prefixed snapshot (written by preprocess()) over the plain
    column name, since the plain column may have been MinMax-scaled for
    ML-feature purposes by the time this row reaches assign_token().
    """
    for name in names:
        raw_name = RAW_COL_PREFIX + name
        if raw_name in row.index:
            val = row[raw_name]
        elif name in row.index:
            val = row[name]
        else:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            return default
    return default


def assign_token(row: pd.Series) -> str:
    """Rule-based mapping from a flow row to one of the 10 VOCAB tokens."""
    dst_port = int(get_col(row, COL_VARIANTS["dst_port"], 0))
    duration_ms = get_col(row, COL_VARIANTS["duration"], 0.0)
    duration_s = max(duration_ms / 1e6, 1e-6)  # CIC-IDS2017 duration is in microseconds

    fwd_pkts = get_col(row, COL_VARIANTS["total_fwd_pkts"], 0.0)
    bwd_pkts = get_col(row, COL_VARIANTS["total_bwd_pkts"], 0.0)
    total_pkts = fwd_pkts + bwd_pkts

    fwd_bytes = get_col(row, COL_VARIANTS["total_fwd_bytes"], 0.0)
    bwd_bytes = get_col(row, COL_VARIANTS["total_bwd_bytes"], 0.0)
    total_bytes = fwd_bytes + bwd_bytes

    syn = get_col(row, COL_VARIANTS["syn"], 0.0)
    ack = get_col(row, COL_VARIANTS["ack"], 0.0)
    fin = get_col(row, COL_VARIANTS["fin"], 0.0)
    rst = get_col(row, COL_VARIANTS["rst"], 0.0)

    pkt_rate = total_pkts / duration_s

    if pkt_rate > 10000 or (total_pkts > 500 and duration_s < 1):
        return "DOS_INDICATOR"
    if syn > 0 and ack == 0 and total_bytes < 100:
        return "SCAN_ACTIVITY"
    if rst > 0 and total_bytes < 500:
        return "FORCED_TERMINATION"
    if dst_port in AUTH_PORTS and rst > 0 and total_bytes < 2000:
        return "AUTH_FAIL"
    if dst_port in AUTH_PORTS and fwd_pkts > 2 and bwd_pkts > 2 and rst == 0:
        return "AUTH_SUCCESS"
    if bwd_bytes > 5000 and fwd_pkts > 1:
        return "POST_AUTH_ACTIVITY"
    if fwd_pkts > 1 and bwd_pkts > 1 and total_bytes > 1000:
        return "DATA_TRANSFER"
    if syn > 0 and ack > 0 and fin == 0 and rst == 0:
        return "SESSION_ESTABLISHED"
    if fin > 0:
        return "CONNECTION_TERMINATION"
    return "CONNECTION_ATTEMPT"


def _session_label(labels: pd.Series) -> str:
    """
    Attack-priority session labeling: if ANY flow in the session is
    non-BENIGN, the session is labeled with the most frequent non-BENIGN
    label among its flows; only an all-BENIGN session is labeled BENIGN.

    A pure majority vote (the previous behavior) erases a session's
    attack label whenever the attack flows are a minority within it —
    e.g. a multi-flow session with a few malicious packets buried among
    many benign-looking handshake/control flows would get mislabeled
    BENIGN outright. A session containing any malicious flow is
    malicious; that is the standard convention for session-level NIDS
    labeling and it doesn't require majority representation.
    """
    non_benign = labels[labels != "BENIGN"]
    if len(non_benign) > 0:
        return non_benign.mode().iat[0]
    return "BENIGN"


def encode_sessions(df_sess: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Apply assign_token to every row and group tokens by SessionID."""
    df = df_sess.copy()
    df["Token"] = df.apply(assign_token, axis=1)
    df["TokenID"] = df["Token"].map(TOKEN2ID)

    sessions_dict: Dict[str, dict] = {}
    for session_id, group in df.groupby("SessionID"):
        tokens = group["Token"].tolist()
        token_ids = group["TokenID"].tolist()
        label = _session_label(group["Label"]) if "Label" in group.columns else "BENIGN"
        sessions_dict[session_id] = {
            "tokens": tokens,
            "token_ids": token_ids,
            "label": label,
        }
    return df, sessions_dict


def sessions_to_arrays(
    sessions_dict: Dict[str, dict], le: LabelEncoder, max_len: int = MAX_SEQ_LEN
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad/truncate token-id sequences to max_len; map labels through le."""
    known = set(le.classes_)
    session_ids = list(sessions_dict.keys())
    n = len(session_ids)

    X_seqs = np.zeros((n, max_len), dtype=np.int64)
    y_seqs = np.zeros(n, dtype=np.int64)

    for i, sid in enumerate(session_ids):
        token_ids = sessions_dict[sid]["token_ids"][:max_len]
        X_seqs[i, : len(token_ids)] = token_ids

        label = sessions_dict[sid]["label"]
        if label in known:
            y_seqs[i] = le.transform([label])[0]
        else:
            y_seqs[i] = 0

    return X_seqs, y_seqs


def compute_sample_weights_lstm(y_seqs: np.ndarray, class_weights: Dict[int, float]) -> np.ndarray:
    """Map each sequence's class id to its precomputed class weight."""
    return np.array([class_weights.get(int(y), 1.0) for y in y_seqs], dtype=np.float32)
