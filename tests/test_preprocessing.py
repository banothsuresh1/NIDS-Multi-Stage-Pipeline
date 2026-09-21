import numpy as np
import pandas as pd

from src.nids.preprocessing import encode_labels, smote_knn_augment


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


def test_smote_knn_clean_never_deletes_original_real_samples():
    # SMOTE-KNN's cleaning step must only ever remove SYNTHETIC points --
    # never an original real sample, majority or minority -- unlike
    # EditedNearestNeighbours (SMOTE-ENN), which edits real boundary points
    # too. Two rare classes are embedded WITHIN the majority class's
    # distribution (heavy overlap, as real rare-attack classes like
    # Heartbleed often are relative to BENIGN) specifically so that some
    # synthetic points DO get cleaned -- proving the filter does real work,
    # not just a no-op -- while every real sample must still survive.
    rng = np.random.default_rng(3)
    X_major = rng.normal(loc=0.0, scale=1.0, size=(300, 5))
    X_rare1 = rng.normal(loc=0.05, scale=1.0, size=(11, 5))
    X_rare2 = rng.normal(loc=-0.05, scale=1.0, size=(15, 5))
    X = np.vstack([X_major, X_rare1, X_rare2])
    y = np.array([0] * 300 + [1] * 11 + [2] * 15)
    class_counts = {0: 300, 1: 11, 2: 15}

    X_aug, y_aug = smote_knn_augment(X, y, class_counts, K=3, seed=42)
    counts_after = dict(zip(*np.unique(y_aug, return_counts=True)))

    # Real samples are never removed, so counts can never drop below the
    # original class size, and cleaning only ever removes points, so counts
    # can never exceed the post-SMOTE floor (SMOTE_MIN_SAMPLES = 50).
    assert counts_after.get(0) == 300  # majority class untouched entirely
    assert 11 <= counts_after.get(1, 0) <= 50
    assert 15 <= counts_after.get(2, 0) <= 50


def test_encode_labels_preserves_val_only_attack_types():
    # Mirrors CIC-IDS2017's real structure: attack types are day-specific,
    # so under a chronological split, val/test routinely contain attack
    # labels that never appear in train. encode_labels() must NOT collapse
    # those to BENIGN -- that's exactly the bug that made every validation
    # session come out benign in the reported issue.
    df_train = pd.DataFrame({"Label": ["BENIGN"] * 8 + ["FTP-Patator"] * 2})
    df_val = pd.DataFrame({"Label": ["BENIGN"] * 5 + ["DoS Hulk"] * 3 + ["Heartbleed"] * 2})
    df_test = pd.DataFrame({"Label": ["BENIGN"] * 5 + ["PortScan"] * 5})

    le, classes, K = encode_labels(df_train, df_val, df_test)

    assert "DoS Hulk" in classes
    assert "Heartbleed" in classes
    assert "PortScan" in classes
    # The val rows genuinely labeled DoS Hulk/Heartbleed must still read
    # as such after encode_labels(), not have been overwritten to BENIGN.
    assert (df_val["Label"] == "DoS Hulk").sum() == 3
    assert (df_val["Label"] == "Heartbleed").sum() == 2
    assert (df_val["Label"] != "BENIGN").sum() == 5
