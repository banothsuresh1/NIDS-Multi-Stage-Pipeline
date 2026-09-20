import pandas as pd

from config import MAX_SEQ_LEN, VOCAB
from src.nids.events import assign_token, encode_sessions, sessions_to_arrays


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
