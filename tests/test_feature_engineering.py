from config import GROUP_A_SIZE
from src.nids.feature_engineering import (
    assign_feature_groups,
    compute_mi_scores,
    get_model_features,
)


def _groups(preprocessed_data):
    d = preprocessed_data
    mi_scores = compute_mi_scores(d["train"][d["feat_cols"]], d["y_train_A"], d["feat_cols"], 42)
    return assign_feature_groups(d["feat_cols"], mi_scores)


def test_group_a_size(preprocessed_data):
    group_a, group_b, group_c, group_d = _groups(preprocessed_data)
    assert len(group_a) <= GROUP_A_SIZE
    assert len(group_a) <= len(preprocessed_data["feat_cols"])


def test_all_groups_non_empty(preprocessed_data):
    for group in _groups(preprocessed_data):
        assert len(group) > 0


def test_group_a_and_group_c_disjoint(preprocessed_data):
    group_a, _, group_c, _ = _groups(preprocessed_data)
    assert set(group_a).isdisjoint(set(group_c))


def test_rf_features_is_union_of_a_and_b(preprocessed_data):
    group_a, group_b, group_c, group_d = _groups(preprocessed_data)
    rf_features, _, _ = get_model_features(group_a, group_b, group_c, group_d)
    assert set(rf_features) == set(group_a) | set(group_b)
    assert len(rf_features) == len(set(rf_features))  # deduplicated


def test_xgb_features_is_union_of_b_and_d(preprocessed_data):
    group_a, group_b, group_c, group_d = _groups(preprocessed_data)
    _, _, xgb_features = get_model_features(group_a, group_b, group_c, group_d)
    assert set(xgb_features) == set(group_b) | set(group_d)
    assert len(xgb_features) == len(set(xgb_features))
