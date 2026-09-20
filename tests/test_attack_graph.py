import pytest

from config import NOVEL_EDGE_WEIGHT, VOCAB
from src.nids.attack_graph import (
    build_attack_graph,
    compute_tc_score,
    update_graph_ema,
)


def test_graph_has_all_vocab_nodes(attack_graph_fixture):
    assert attack_graph_fixture.number_of_nodes() == len(VOCAB)
    assert set(attack_graph_fixture.nodes()) == set(VOCAB)


def test_edge_weights_in_range(attack_graph_fixture):
    for _, _, data in attack_graph_fixture.edges(data=True):
        assert 0 < data["weight"] <= 1


def test_ema_update_moves_toward_new_observation():
    sessions_batch1 = [{"tokens": ["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED"]}] * 10
    G = build_attack_graph({f"s{i}": s for i, s in enumerate(sessions_batch1)})
    w_before = G["CONNECTION_ATTEMPT"]["SESSION_ESTABLISHED"]["weight"]

    sessions_batch2 = [{"tokens": ["CONNECTION_ATTEMPT", "DOS_INDICATOR"]}] * 10
    G = update_graph_ema(G, sessions_batch2, rho=0.9)
    w_after = G["CONNECTION_ATTEMPT"]["SESSION_ESTABLISHED"]["weight"]

    # existing edge should shrink toward 0 (not replaced outright) since it
    # received no new observations in batch2
    assert w_after < w_before
    assert w_after > 0


def test_novel_edges_get_novel_weight():
    import math

    from config import LAMBDA_DECAY

    G = build_attack_graph({"s0": {"tokens": ["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED"]}})
    score = compute_tc_score(["SCAN_ACTIVITY", "AUTH_FAIL"], G)
    # SCAN_ACTIVITY->AUTH_FAIL is not in G -> should use NOVEL_EDGE_WEIGHT,
    # decayed by exp(-lam * delta_t) with the default delta_t=1.0
    expected = NOVEL_EDGE_WEIGHT * math.exp(-LAMBDA_DECAY)
    assert score == pytest.approx(expected, rel=1e-6)


def test_tc_score_zero_for_single_token(attack_graph_fixture):
    assert compute_tc_score(["CONNECTION_ATTEMPT"], attack_graph_fixture) == 0.0


def test_tc_score_range_for_normal_session(attack_graph_fixture):
    score = compute_tc_score(
        ["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED", "DATA_TRANSFER"], attack_graph_fixture
    )
    assert 0.0 <= score <= 1.0
