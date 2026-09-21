import pandas as pd

from config import SESSION_TIMEOUT
from src.nids.sessions import make_session_key, reconstruct_sessions, stratified_group_session_split


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


def _sessions_dict(label_counts):
    """Build a minimal sessions_dict: {label: n_sessions} -> {sid: {"label": ...}}."""
    sessions = {}
    i = 0
    for label, n in label_counts.items():
        for _ in range(n):
            sessions[f"s{i}"] = {"tokens": ["CONNECTION_ATTEMPT"], "token_ids": [1], "label": label}
            i += 1
    return sessions


def test_stratified_group_session_split_partitions_disjoint():
    sessions = _sessions_dict({"BENIGN": 200, "FTP-Patator": 60, "SSH-Patator": 50})
    train_ids, val_ids, test_ids = stratified_group_session_split(sessions, val_size=0.15, test_size=0.15)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == set(sessions.keys())


def test_stratified_group_session_split_every_well_represented_class_in_all_three():
    sessions = _sessions_dict({"BENIGN": 300, "FTP-Patator": 90, "SSH-Patator": 60, "Bot": 30})
    train_ids, val_ids, test_ids = stratified_group_session_split(sessions, val_size=0.15, test_size=0.15)

    def _labels(ids):
        return {sessions[sid]["label"] for sid in ids}

    # Every class here has far more than 3 sessions, so all must appear
    # in all three partitions -- this is the exact property the split
    # must guarantee (a plain two-stage stratified split can leave a
    # rare-ish class out of one side purely by rounding).
    assert _labels(train_ids) == _labels(val_ids) == _labels(test_ids) == {"BENIGN", "FTP-Patator", "SSH-Patator", "Bot"}


def test_stratified_group_session_split_guarantees_rare_class_in_all_three():
    # A class with exactly 3 sessions is the minimum for which 3-way
    # coverage is even mathematically possible; it must get exactly one
    # session in each partition.
    sessions = _sessions_dict({"BENIGN": 100, "Heartbleed": 3})
    train_ids, val_ids, test_ids = stratified_group_session_split(sessions, val_size=0.15, test_size=0.15)

    heartbleed_by_partition = [
        sum(1 for sid in ids if sessions[sid]["label"] == "Heartbleed")
        for ids in (train_ids, val_ids, test_ids)
    ]
    assert heartbleed_by_partition == [1, 1, 1]


def test_stratified_group_session_split_undersized_class_reported_not_crashed():
    # A class with fewer than 3 total sessions cannot possibly appear in
    # all three partitions -- the function must not raise (unlike a raw
    # sklearn stratified split, which errors on classes with <2 members)
    # and must still place every session somewhere.
    sessions = _sessions_dict({"BENIGN": 100, "Heartbleed": 2})
    train_ids, val_ids, test_ids = stratified_group_session_split(sessions, val_size=0.15, test_size=0.15)
    assert train_ids | val_ids | test_ids == set(sessions.keys())
