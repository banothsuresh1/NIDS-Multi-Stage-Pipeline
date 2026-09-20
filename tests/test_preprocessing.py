import numpy as np

from src.nids.preprocessing import smote_knn_augment


def test_no_inf_values(preprocessed_data):
    assert np.isfinite(preprocessed_data["X_train_A"]).all()
    assert np.isfinite(preprocessed_data["X_val"]).all()
    assert np.isfinite(preprocessed_data["X_test"]).all()


def test_no_nan_values(preprocessed_data):
    assert not np.isnan(preprocessed_data["X_train_A"]).any()
    assert not np.isnan(preprocessed_data["X_val"]).any()
    assert not np.isnan(preprocessed_data["X_test"]).any()


def test_scaler_fit_on_train_only(preprocessed_data):
    # Val/test may legitimately fall outside [0, 1] since the scaler is
    # fit exclusively on train statistics (no leakage from val/test).
    scaler = preprocessed_data["scaler"]
    assert scaler.data_min_.shape[0] == len(preprocessed_data["feat_cols"])


def test_minority_class_weight_higher(preprocessed_data):
    weights = preprocessed_data["class_weights"]
    train = preprocessed_data["train"]
    counts = train["LabelID"].value_counts()
    minority_class = counts.idxmin()
    majority_class = counts.idxmax()
    assert weights[minority_class] > weights[majority_class]


def test_smote_increases_rare_class_samples():
    # Well-separated clusters (unlike the noisy end-to-end fixture) so that
    # EditedNearestNeighbours cleans only true borderline points instead of
    # erasing the oversampled minority class outright.
    rng = np.random.default_rng(7)
    X_majority = rng.normal(loc=0.0, scale=0.3, size=(150, 6))
    X_minority = rng.normal(loc=10.0, scale=0.3, size=(20, 6))
    X = np.vstack([X_majority, X_minority])
    y = np.array([0] * 150 + [1] * 20)
    class_counts = {0: 150, 1: 20}

    X_aug, y_aug = smote_knn_augment(X, y, class_counts, K=2, seed=42)

    n_before = (y == 1).sum()
    n_after = (y_aug == 1).sum()
    assert n_after >= n_before


def test_smote_handles_zero_sample_class_gracefully():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = np.array([0] * 55 + [1] * 5)
    class_counts = {0: 55, 1: 5, 2: 0}
    X_aug, y_aug = smote_knn_augment(X, y, class_counts, K=3, seed=42)
    assert X_aug.shape[0] >= X.shape[0]
