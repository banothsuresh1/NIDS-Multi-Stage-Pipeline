import pandas as pd

from config import SESSION_TIMEOUT
from src.nids.sessions import make_session_key, reconstruct_sessions


def test_session_key_symmetric():
    row_fwd = pd.Series({
        " Source IP": "10.0.0.1", " Destination IP": "10.0.1.5",
        " Source Port": 5000, " Destination Port": 443, " Protocol": 6,
    })
    row_rev = pd.Series({
        " Source IP": "10.0.1.5", " Destination IP": "10.0.0.1",
        " Source Port": 443, " Destination Port": 5000, " Protocol": 6,
    })
    assert make_session_key(row_fwd) == make_session_key(row_rev)


def _two_flow_df(gap_seconds: float) -> pd.DataFrame:
    base_ts = pd.Timestamp("2017-07-03 09:00:00")
    return pd.DataFrame({
        "Source IP": ["10.0.0.1", "10.0.0.1"],
        "Destination IP": ["10.0.1.5", "10.0.1.5"],
        "Source Port": [5000, 5000],
        "Destination Port": [443, 443],
        "Protocol": [6, 6],
        "Timestamp": [base_ts, base_ts + pd.Timedelta(seconds=gap_seconds)],
        "Label": ["BENIGN", "BENIGN"],
    })


def test_gap_greater_than_timeout_creates_new_session():
    df = _two_flow_df(SESSION_TIMEOUT + 30)
    df_sess = reconstruct_sessions(df, timeout=SESSION_TIMEOUT)
    assert df_sess["SessionID"].nunique() == 2


def test_gap_less_than_timeout_same_session():
    df = _two_flow_df(SESSION_TIMEOUT - 10)
    df_sess = reconstruct_sessions(df, timeout=SESSION_TIMEOUT)
    assert df_sess["SessionID"].nunique() == 1


def test_no_nan_session_ids():
    df = _two_flow_df(5)
    df_sess = reconstruct_sessions(df)
    assert df_sess["SessionID"].notna().all()


def test_single_flow_session():
    df = _two_flow_df(5).iloc[[0]].reset_index(drop=True)
    df_sess = reconstruct_sessions(df)
    assert df_sess["SessionID"].nunique() == 1
    assert len(df_sess) == 1


def test_reconstruct_sessions_on_full_fixture(synthetic_split):
    df_train, _, _ = synthetic_split
    df_sess = reconstruct_sessions(df_train)
    assert "SessionID" in df_sess.columns
    assert df_sess["SessionID"].notna().all()
    assert df_sess["SessionID"].nunique() > 0
    assert len(df_sess) == len(df_train)


def test_session_id_prefix_prevents_cross_split_collisions(synthetic_split):
    # reconstruct_sessions()'s id counter restarts at 1 on every call, so
    # two separate calls (e.g. one per train/val/test split, as the
    # notebook does) would otherwise produce identical SessionID strings
    # for entirely different sessions.
    df_train, df_val, _ = synthetic_split
    df_train_sess = reconstruct_sessions(df_train, session_id_prefix="TRAIN_")
    df_val_sess = reconstruct_sessions(df_val, session_id_prefix="VAL_")

    train_ids = set(df_train_sess["SessionID"])
    val_ids = set(df_val_sess["SessionID"])
    assert train_ids.isdisjoint(val_ids)
    assert all(sid.startswith("TRAIN_") for sid in train_ids)
    assert all(sid.startswith("VAL_") for sid in val_ids)
