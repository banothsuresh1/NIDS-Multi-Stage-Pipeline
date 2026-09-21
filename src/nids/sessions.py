"""
Stage 4 — Bidirectional Session Reconstruction.

Groups individual (uni-directional) flow records into bidirectional network
sessions keyed on a symmetric 5-tuple, with timeout-based session splitting
so that a long-idle re-use of the same 5-tuple starts a new session.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from config import SESSION_TIMEOUT

SRC_IP_VARIANTS = ["Src IP", " Source IP", "Source IP", "Src_IP"]
DST_IP_VARIANTS = ["Dst IP", " Destination IP", "Destination IP", "Dst_IP"]
SRC_PORT_VARIANTS = ["Src Port", " Source Port", "Source Port"]
DST_PORT_VARIANTS = ["Dst Port", " Destination Port", "Destination Port"]
PROTOCOL_VARIANTS = ["Protocol", " Protocol"]
TIMESTAMP_VARIANTS = ["Timestamp", " Timestamp", "timestamp"]


def _first_present(df: pd.DataFrame, variants) -> str | None:
    for name in variants:
        if name in df.columns:
            return name
    return None


def _value_for_variant(row: pd.Series, variants, default=0):
    for name in variants:
        if name in row.index:
            return row[name]
    return default


def make_session_key(row: pd.Series) -> tuple:
    """
    Symmetric 5-tuple key: order-independent so A->B and B->A collide.
    Auto-detects whichever CIC-IDS2017 column-name variant is present
    (e.g. "Src IP" vs " Source IP", "Dst Port" vs " Destination Port").
    """
    src_ip = _value_for_variant(row, SRC_IP_VARIANTS, "")
    dst_ip = _value_for_variant(row, DST_IP_VARIANTS, "")
    src_port = _value_for_variant(row, SRC_PORT_VARIANTS, 0)
    dst_port = _value_for_variant(row, DST_PORT_VARIANTS, 0)
    protocol = _value_for_variant(row, PROTOCOL_VARIANTS, 0)

    ip_lo, ip_hi = sorted([str(src_ip), str(dst_ip)])
    try:
        port_lo, port_hi = sorted([float(src_port), float(dst_port)])
    except (TypeError, ValueError):
        port_lo, port_hi = 0.0, 0.0

    return (ip_lo, ip_hi, protocol, port_lo, port_hi)


def reconstruct_sessions(
    df: pd.DataFrame, timeout: int = SESSION_TIMEOUT, session_id_prefix: str = ""
) -> pd.DataFrame:
    """
    Assign SessionID to each flow row via symmetric-key + timeout grouping.

    session_id_prefix should be distinct per call (e.g. "TRAIN_", "VAL_",
    "TEST_") whenever reconstruct_sessions() is called separately per
    split, as the notebook does: the id counter restarts at 1 on every
    call, so without a prefix train's SESS_00000001 and val's
    SESS_00000001 are literally the same string despite being different
    sessions. That doesn't corrupt the current pipeline (each split's
    sessions live in separate dicts throughout), but SessionID is also
    surfaced directly to analysts (explainability.build_evidence_report),
    where a collision would misleadingly suggest two different sessions
    from two different days are the same one.
    """
    df = df.copy()

    ts_col = _first_present(df, TIMESTAMP_VARIANTS)
    df["_key"] = df.apply(make_session_key, axis=1)

    if ts_col is not None:
        df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
        # Rows with unparseable timestamps fall back to row-order pseudo-time.
        if df["_ts"].isna().all():
            df["_ts"] = pd.to_datetime(np.arange(len(df)), unit="s", origin="unix")
        else:
            fallback_ts = pd.Series(
                pd.to_datetime(np.arange(len(df)), unit="s", origin="unix"), index=df.index
            )
            df["_ts"] = df["_ts"].fillna(fallback_ts)
    else:
        # No timestamp column found: assume rows are 1 second apart in order.
        df["_ts"] = pd.to_datetime(np.arange(len(df)), unit="s", origin="unix")

    df["_orig_order"] = np.arange(len(df))
    df = df.sort_values(["_key", "_ts", "_orig_order"]).reset_index(drop=True)

    session_ids = np.empty(len(df), dtype=object)
    current_id = 0
    prev_key = None
    prev_ts = None

    for i in range(len(df)):
        key = df.at[i, "_key"]
        ts = df.at[i, "_ts"]
        if prev_key is None or key != prev_key:
            current_id += 1
        elif pd.notna(ts) and pd.notna(prev_ts):
            gap = (ts - prev_ts).total_seconds()
            if gap > timeout:
                current_id += 1
        session_ids[i] = f"{session_id_prefix}SESS_{current_id:08d}"
        prev_key = key
        prev_ts = ts

    df["SessionID"] = session_ids
    df = df.sort_values("_orig_order").reset_index(drop=True)
    df = df.drop(columns=["_key", "_ts", "_orig_order"])
    return df


def get_session_stats(df_sess: pd.DataFrame) -> pd.DataFrame:
    """Per-session summary: flow count, dominant label, approximate duration."""
    ts_col = _first_present(df_sess, TIMESTAMP_VARIANTS)

    def _agg(group: pd.DataFrame) -> pd.Series:
        n_flows = len(group)
        label = group["Label"].mode().iat[0] if "Label" in group.columns else None
        if ts_col is not None:
            ts = pd.to_datetime(group[ts_col], errors="coerce").dropna()
            duration = (ts.max() - ts.min()).total_seconds() if len(ts) > 1 else 0.0
        else:
            duration = float(n_flows - 1)
        return pd.Series({"n_flows": n_flows, "label": label, "duration": duration})

    stats = df_sess.groupby("SessionID").apply(_agg).reset_index()
    stats = stats.rename(columns={"SessionID": "session_id"})
    return stats
