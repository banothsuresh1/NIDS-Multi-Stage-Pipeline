import numpy as np

from src.nids.attack_graph import build_attack_graph
from src.nids.explainability import (
    build_evidence_report,
    compute_treeshap_rf,
    extract_graph_path_evidence,
)


def test_compute_treeshap_rf_shape(trained_rf, preprocessed_data):
    d = preprocessed_data
    X_sample = d["X_test"][:20]
    explainer, shap_values = compute_treeshap_rf(trained_rf, X_sample, d["feat_cols"])
    if isinstance(shap_values, list):
        assert shap_values[0].shape[0] == X_sample.shape[0]
    else:
        assert shap_values.shape[0] == X_sample.shape[0]


def test_extract_graph_path_evidence_keys(attack_graph_fixture):
    tokens = ["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED", "DATA_TRANSFER"]
    evidence = extract_graph_path_evidence(tokens, attack_graph_fixture)
    assert len(evidence) == len(tokens) - 1
    for e in evidence:
        assert set(e.keys()) == {"from", "to", "weight", "novel"}


def test_novel_flag_true_for_unseen_transition():
    G = build_attack_graph({"s0": {"tokens": ["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED"]}})
    evidence = extract_graph_path_evidence(["SCAN_ACTIVITY", "AUTH_FAIL"], G)
    assert evidence[0]["novel"] is True


def test_novel_flag_false_for_seen_transition():
    G = build_attack_graph({"s0": {"tokens": ["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED"]}})
    evidence = extract_graph_path_evidence(["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED"], G)
    assert evidence[0]["novel"] is False


def test_build_evidence_report_non_empty(trained_rf, preprocessed_data, attack_graph_fixture):
    d = preprocessed_data
    X_sample = d["X_test"][:20]
    _, shap_values = compute_treeshap_rf(trained_rf, X_sample, d["feat_cols"])

    session_info = {"tokens": ["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED", "DATA_TRANSFER"], "label": "DoS Hulk"}
    path_evidence = extract_graph_path_evidence(session_info["tokens"], attack_graph_fixture)

    report = build_evidence_report(
        "SESS_TEST_001", session_info, risk_score=0.82, sp_score=0.5, tc_score=0.6, gw_score=0.3,
        shap_values_rf=shap_values, rf_feature_names=d["feat_cols"], path_evidence=path_evidence,
    )
    assert isinstance(report, str)
    assert len(report) > 0
    assert "SESS_TEST_001" in report
