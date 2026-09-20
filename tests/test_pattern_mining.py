from src.nids.pattern_mining import (
    build_transaction_db,
    compute_sp_score,
    run_fpgrowth,
    run_prefixspan,
)


def test_build_transaction_db_boolean_columns(synthetic_sessions):
    sessions_train, _, _ = synthetic_sessions
    te, df_trans = build_transaction_db(sessions_train)
    assert df_trans.shape[0] == len(sessions_train)
    assert df_trans.dtypes.apply(lambda dt: dt == bool).all()


def test_run_fpgrowth_columns(synthetic_sessions):
    sessions_train, _, _ = synthetic_sessions
    _, df_trans = build_transaction_db(sessions_train)
    fp_patterns = run_fpgrowth(df_trans, min_support=0.01)
    assert "support" in fp_patterns.columns
    assert "itemsets" in fp_patterns.columns


def test_compute_sp_score_range(synthetic_sessions):
    sessions_train, _, _ = synthetic_sessions
    _, df_trans = build_transaction_db(sessions_train)
    fp_patterns = run_fpgrowth(df_trans, min_support=0.01)
    ps_patterns = run_prefixspan(sessions_train, min_support=0.01, max_gap=8)

    any_session = next(iter(sessions_train.values()))
    score = compute_sp_score(any_session["tokens"], fp_patterns, ps_patterns)
    assert 0.0 <= score <= 1.0


def test_run_prefixspan_returns_list(synthetic_sessions):
    sessions_train, _, _ = synthetic_sessions
    ps_patterns = run_prefixspan(sessions_train, min_support=0.01, max_gap=8)
    assert isinstance(ps_patterns, list)


def test_sp_score_higher_for_matching_pattern():
    sessions = {
        f"s{i}": {"tokens": ["SCAN_ACTIVITY", "DOS_INDICATOR"], "token_ids": [], "label": "DoS"}
        for i in range(10)
    }
    sessions["benign"] = {"tokens": ["CONNECTION_ATTEMPT"], "token_ids": [], "label": "BENIGN"}

    _, df_trans = build_transaction_db(sessions)
    fp_patterns = run_fpgrowth(df_trans, min_support=0.1)
    ps_patterns = run_prefixspan(sessions, min_support=0.1, max_gap=8)

    matching_score = compute_sp_score(["SCAN_ACTIVITY", "DOS_INDICATOR"], fp_patterns, ps_patterns)
    non_matching_score = compute_sp_score(["CONNECTION_ATTEMPT"], fp_patterns, ps_patterns)
    assert matching_score >= non_matching_score
