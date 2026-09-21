import numpy as np
import pandas as pd

from config import MAX_SEQ_LEN, PAD_ID, TOKEN2ID, VOCAB, VOCAB_SIZE
from src.nids.events import RAW_COL_PREFIX, assign_token, encode_sessions, get_col, sessions_to_arrays


def test_no_real_token_maps_to_pad_id():
    # id 0 is reserved for padding (sessions_to_arrays zero-pads, and
    # build_lstm's Embedding uses mask_zero=True); a real token colliding
    # with it would be silently masked out identically to padding.
    assert PAD_ID not in TOKEN2ID.values()
    assert VOCAB_SIZE == len(VOCAB) + 1


def test_get_col_prefers_raw_snapshot_over_scaled_value():
    # assign_token's thresholds are written in raw units. If a row carries
    # both the (possibly MinMax-scaled) plain column and preprocess()'s
    # "__raw__"-prefixed unscaled snapshot, get_col must read the raw
    # snapshot -- otherwise every threshold silently breaks once the
    # pipeline's Stage 2 output feeds Stage 5 (which is the real call
    # pattern in the notebook).
    row = pd.Series({
        "Flow Duration": 0.13,  # a plausible MinMax-scaled value, NOT raw microseconds
        RAW_COL_PREFIX + "Flow Duration": 500_000.0,  # the true raw value
    })
    assert get_col(row, ["Flow Duration"]) == 500_000.0


def test_get_col_falls_back_to_plain_column_when_no_raw_snapshot():
    row = pd.Series({"Flow Duration": 500_000.0})
    assert get_col(row, ["Flow Duration"]) == 500_000.0


def test_assign_token_uses_raw_magnitudes_not_scaled_ones():
    # Reproduces the actual failure mode: a row whose flow-statistic
    # columns have been MinMax-scaled to [0, 1] (as preprocess() does for
    # the ML feature set) would make assign_token's duration_s collapse
    # toward ~0 and pkt_rate explode, firing DOS_INDICATOR unconditionally
    # if it read the scaled columns. With the raw snapshot present, it
    # must classify by the true magnitude instead -- here, a normal
    # bidirectional exchange with no attack signature.
    row = pd.Series({
        "Destination Port": 8080,
        RAW_COL_PREFIX + "Flow Duration": 500_000.0,      # scaled equivalent might be ~0.13
        "Flow Duration": 0.13,
        RAW_COL_PREFIX + "Total Fwd Packets": 3.0,
        "Total Fwd Packets": 0.05,
        RAW_COL_PREFIX + "Total Backward Packets": 3.0,
        "Total Backward Packets": 0.05,
        RAW_COL_PREFIX + "Total Length of Fwd Packets": 200.0,
        "Total Length of Fwd Packets": 0.02,
        RAW_COL_PREFIX + "Total Length of Bwd Packets": 200.0,
        "Total Length of Bwd Packets": 0.02,
        RAW_COL_PREFIX + "SYN Flag Count": 0.0,
        "SYN Flag Count": 0.0,
        RAW_COL_PREFIX + "ACK Flag Count": 0.0,
        "ACK Flag Count": 0.0,
        RAW_COL_PREFIX + "FIN Flag Count": 0.0,
        "FIN Flag Count": 0.0,
        RAW_COL_PREFIX + "RST Flag Count": 0.0,
        "RST Flag Count": 0.0,
    })
    assert assign_token(row) != "DOS_INDICATOR"


def _row(**overrides):
    base = {
        "Destination Port": 8080,
        "Flow Duration": 500_000,
        "Total Fwd Packets": 3,
        "Total Backward Packets": 3,
        "Total Length of Fwd Packets": 200,
        "Total Length of Bwd Packets": 200,
        "SYN Flag Count": 0,
        "ACK Flag Count": 0,
        "FIN Flag Count": 0,
        "RST Flag Count": 0,
    }
    base.update(overrides)
    return pd.Series(base)


def test_assign_token_returns_valid_vocab_token():
    for _ in range(5):
        token = assign_token(_row())
        assert token in VOCAB


def test_dos_indicator_high_pkt_rate():
    row = _row(
        **{
            "Flow Duration": 1,  # microseconds -> ~1e-6s duration
            "Total Fwd Packets": 5000,
            "Total Backward Packets": 5000,
        }
    )
    assert assign_token(row) == "DOS_INDICATOR"


def test_auth_fail_ssh_port_with_rst():
    # total_bytes must be >= 500 (else FORCED_TERMINATION takes priority)
    # and < 2000 to satisfy the AUTH_FAIL rule.
    row = _row(
        **{
            "Destination Port": 22,
            "RST Flag Count": 1,
            "Total Length of Fwd Packets": 800,
            "Total Length of Bwd Packets": 800,
        }
    )
    assert assign_token(row) == "AUTH_FAIL"


def test_auth_success_ssh_port_bidirectional():
    row = _row(
        **{
            "Destination Port": 22,
            "Total Fwd Packets": 10,
            "Total Backward Packets": 10,
            "RST Flag Count": 0,
        }
    )
    assert assign_token(row) == "AUTH_SUCCESS"


def test_encode_sessions_structure(synthetic_sessions):
    sessions_train, _, _ = synthetic_sessions
    assert len(sessions_train) > 0
    for sid, info in sessions_train.items():
        assert "tokens" in info and "token_ids" in info and "label" in info
        assert len(info["tokens"]) == len(info["token_ids"])
        break


def test_session_label_is_attack_priority_not_majority_vote():
    # A session where the attack flow is a MINORITY (1 attack vs 4 benign)
    # must still be labeled with the attack, not BENIGN. A pure majority
    # vote would erase the attack label whenever it's outnumbered within
    # its own session -- exactly the bug that made every validation
    # session come out BENIGN in the reported issue.
    df = pd.DataFrame({
        "SessionID": ["s1"] * 5,
        "Label": ["BENIGN", "BENIGN", "BENIGN", "BENIGN", "DoS Hulk"],
        "Destination Port": [80] * 5,
        "Flow Duration": [500_000] * 5,
        "Total Fwd Packets": [3] * 5,
        "Total Backward Packets": [3] * 5,
        "Total Length of Fwd Packets": [200] * 5,
        "Total Length of Bwd Packets": [200] * 5,
        "SYN Flag Count": [0] * 5,
        "ACK Flag Count": [0] * 5,
        "FIN Flag Count": [0] * 5,
        "RST Flag Count": [0] * 5,
    })
    _, sessions_dict = encode_sessions(df)
    assert sessions_dict["s1"]["label"] == "DoS Hulk"


def test_session_label_all_benign_stays_benign():
    df = pd.DataFrame({
        "SessionID": ["s1"] * 3,
        "Label": ["BENIGN"] * 3,
        "Destination Port": [80] * 3,
        "Flow Duration": [500_000] * 3,
        "Total Fwd Packets": [3] * 3,
        "Total Backward Packets": [3] * 3,
        "Total Length of Fwd Packets": [200] * 3,
        "Total Length of Bwd Packets": [200] * 3,
        "SYN Flag Count": [0] * 3,
        "ACK Flag Count": [0] * 3,
        "FIN Flag Count": [0] * 3,
        "RST Flag Count": [0] * 3,
    })
    _, sessions_dict = encode_sessions(df)
    assert sessions_dict["s1"]["label"] == "BENIGN"


def test_sessions_to_arrays_padding():
    sessions_dict = {
        "s1": {"tokens": ["CONNECTION_ATTEMPT"], "token_ids": [0], "label": "BENIGN"},
    }
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(["BENIGN", "DoS Hulk"])
    X, y = sessions_to_arrays(sessions_dict, le, max_len=MAX_SEQ_LEN)
    assert X.shape == (1, MAX_SEQ_LEN)
    assert (X[0, 1:] == 0).all()


def test_sessions_to_arrays_truncation():
    long_tokens = ["CONNECTION_ATTEMPT"] * (MAX_SEQ_LEN + 10)
    long_ids = [0] * (MAX_SEQ_LEN + 10)
    sessions_dict = {"s1": {"tokens": long_tokens, "token_ids": long_ids, "label": "BENIGN"}}
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(["BENIGN"])
    X, y = sessions_to_arrays(sessions_dict, le, max_len=MAX_SEQ_LEN)
    assert X.shape == (1, MAX_SEQ_LEN)


def test_token_diversity_survives_full_preprocess_to_encode_sessions_chain(synthetic_split):
    # Integration regression test for the actual reported failure mode:
    # the real notebook calls reconstruct_sessions()/encode_sessions() on
    # preprocess()'s OUTPUT (Stage 2's MinMax-scaled dataframe), not the
    # raw one. Before the raw-magnitude-snapshot fix, this made
    # assign_token's thresholds fire on already-scaled [0, 1] values,
    # collapsing every session to a single degenerate token. This test
    # chains the real stage order and asserts token diversity survives it.
    from src.nids.preprocessing import preprocess
    from src.nids.sessions import reconstruct_sessions

    df_train, df_val, df_test = synthetic_split
    train, val, test, scaler, feat_cols = preprocess(df_train, df_val, df_test)

    df_train_sess = reconstruct_sessions(train)
    df_train_sess_enc, sessions_train = encode_sessions(df_train_sess)

    token_counts = df_train_sess_enc["Token"].value_counts()
    assert len(token_counts) > 1, (
        "assign_token collapsed to a single token across the whole "
        "(post-preprocess) training set -- it is almost certainly reading "
        "MinMax-scaled values instead of raw magnitudes."
    )
