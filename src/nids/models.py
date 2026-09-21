"""
Stage 6 — Parallel Classifiers.

Three independent classifiers trained on different feature views of the
same sessions/flows:
  Branch A: Random Forest    on GROUP_A U GROUP_B tabular features
  Branch B: BiLSTM            on the behavioral token sequence
  Branch C: XGBoost           on GROUP_B U GROUP_D tabular features
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import (
    LSTM_LR,
    RF_MAX_FEATURES,
    RF_N_ESTIMATORS,
    XGB_COLSAMPLE,
    XGB_LR,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_SUBSAMPLE,
)


def align_proba_to_k(proba: np.ndarray, model_classes: Sequence[int], K: int) -> np.ndarray:
    """
    Re-index a predict_proba output onto the full K-class label space.

    A tree model's classes_ (and hence its predict_proba columns) only
    covers labels actually seen during fit. Extremely rare classes (e.g.
    Heartbleed with N=11 in CIC-IDS2017) can be entirely removed by the
    SMOTE+ENN cleaning step, leaving predict_proba narrower than K. Any
    code that indexes predict_proba columns by global class id (ROC
    curves, evidence fusion) needs the full K-wide, zero-filled array.
    """
    aligned = np.zeros((proba.shape[0], K), dtype=proba.dtype)
    for j, c in enumerate(model_classes):
        aligned[:, int(c)] = proba[:, j]
    return aligned


def train_rf(
    X_train: np.ndarray, y_train: np.ndarray, class_weights: Dict[int, float], seed: int
) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_features=RF_MAX_FEATURES,
        class_weight=class_weights,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X_train, y_train)
    return rf


def build_lstm(vocab_size: int, embed_dim: int, lstm_units: int, dropout: float,
                n_classes: int, lr: float = LSTM_LR):
    from tensorflow import keras
    from tensorflow.keras import layers

    from config import MAX_SEQ_LEN

    model = keras.Sequential([
        layers.Input(shape=(MAX_SEQ_LEN,)),
        layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True),
        layers.Bidirectional(layers.LSTM(lstm_units)),
        layers.Dropout(dropout),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_conv1d(vocab_size: int, embed_dim: int, filters: int, kernel_size: int,
                  dropout: float, n_classes: int, lr: float = LSTM_LR):
    """
    Conv1D + GlobalMaxPooling alternative to the BiLSTM. Sequences here are
    only MAX_SEQ_LEN=20 timesteps over a ~10-token vocabulary, so the
    signal is local motifs (a handful of adjacent tokens), not long-range
    dependencies -- exactly what a small 1D conv is good at, and it
    processes the whole sequence in one parallel pass instead of walking
    it one timestep at a time like an LSTM, which is 10-50x faster on
    CPU for inputs this short. mask_zero is not used here (Conv1D has no
    masking support); padding contributes small, roughly-uniform noise to
    the convolution instead of being explicitly ignored, which is a
    reasonable tradeoff for this speed gain on 20-length sequences.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    from config import MAX_SEQ_LEN

    model = keras.Sequential([
        layers.Input(shape=(MAX_SEQ_LEN,)),
        layers.Embedding(input_dim=vocab_size, output_dim=embed_dim),
        layers.Conv1D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same"),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(dropout),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_sequence_model(vocab_size: int, embed_dim: int, lstm_units: int, dropout: float,
                          n_classes: int, lr: float = LSTM_LR, model_type: str = "bilstm",
                          conv_filters: int = 64, conv_kernel_size: int = 3):
    """Dispatch on config.SEQUENCE_MODEL_TYPE ("bilstm" or "conv1d")."""
    if model_type == "conv1d":
        return build_conv1d(vocab_size, embed_dim, conv_filters, conv_kernel_size,
                             dropout, n_classes, lr)
    return build_lstm(vocab_size, embed_dim, lstm_units, dropout, n_classes, lr)


def train_lstm(model, X_train, y_train, sample_weights, X_val, y_val,
                epochs: int, batch_size: int, patience: int):
    """
    Trains via tf.data (.cache() + .prefetch(AUTOTUNE)) instead of feeding
    raw numpy arrays directly to model.fit(): this overlaps the (trivial
    here, but nonzero) batch-assembly work with the GPU/CPU compute step
    and avoids re-slicing the numpy arrays from scratch every epoch.
    """
    import tensorflow as tf
    from tensorflow import keras

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train, sample_weights))
        .cache()
        .shuffle(buffer_size=min(len(X_train), 10_000), reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop],
        shuffle=False,  # already shuffled via tf.data .shuffle() above
        verbose=2,
    )
    return history


class _GlobalLabelXGBClassifier:
    """
    Wraps an XGBClassifier fit on a locally re-indexed, contiguous label
    space (xgboost's sklearn API requires fit() labels to be exactly
    arange(n_classes_present)) so that predict()/predict_proba()/classes_
    behave in terms of the original GLOBAL class ids.

    This matters whenever SMOTE+ENN removes a class that isn't the last
    one in id order (e.g. Heartbleed, N=11, sitting mid-alphabet) — the
    surviving global ids are then non-contiguous (e.g. {0,1,2,4}), which
    newer xgboost versions reject outright without this remapping.
    """

    def __init__(self, booster, present_classes: np.ndarray):
        self.booster_ = booster  # raw XGBClassifier, e.g. for shap.TreeExplainer
        self.classes_ = present_classes

    def predict(self, X):
        local_pred = self.booster_.predict(X)
        return self.classes_[local_pred]

    def predict_proba(self, X):
        return self.booster_.predict_proba(X)

    def local_index_of(self, global_class_id: int) -> int:
        """Position of a global class id within this model's local (present-classes-only) output columns, or -1 if that class wasn't seen in training."""
        matches = np.flatnonzero(self.classes_ == global_class_id)
        return int(matches[0]) if len(matches) else -1


def train_xgboost(X_train, y_train, class_weights: Dict[int, float], X_val, y_val,
                   n_classes: int, seed: int) -> _GlobalLabelXGBClassifier:
    from xgboost import XGBClassifier

    y_train = np.asarray(y_train)
    present_classes = np.unique(y_train)
    local_of = {int(c): i for i, c in enumerate(present_classes)}
    y_train_local = np.array([local_of[int(y)] for y in y_train])
    sample_weight = np.array([class_weights.get(int(y), 1.0) for y in y_train])

    def _new_clf():
        return XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LR,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )

    # eval_set labels must live in the same local space; any validation row
    # whose label was never seen in training can't be scored there, so it's
    # dropped only for early-stopping/logging purposes.
    y_val_arr = np.asarray(y_val)
    val_mask = np.isin(y_val_arr, present_classes)

    xgb_clf = _new_clf()
    if val_mask.sum() > 0:
        y_val_local = np.array([local_of[int(y)] for y in y_val_arr[val_mask]])
        try:
            xgb_clf.fit(
                X_train, y_train_local,
                sample_weight=sample_weight,
                eval_set=[(np.asarray(X_val)[val_mask], y_val_local)],
                verbose=50,
            )
        except ValueError as exc:
            print(f"[train_xgboost] eval_set rejected ({exc}); fitting without it.")
            xgb_clf = _new_clf()
            xgb_clf.fit(X_train, y_train_local, sample_weight=sample_weight, verbose=50)
    else:
        xgb_clf.fit(X_train, y_train_local, sample_weight=sample_weight, verbose=50)

    return _GlobalLabelXGBClassifier(xgb_clf, present_classes)


def evaluate_classifier(model, X_test, y_test, model_name: str, classes: Sequence[str]) -> dict:
    y_pred = model.predict(X_test)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    labels = list(range(len(classes)))
    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=labels, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(
        y_test, y_pred, labels=labels, target_names=list(classes),
        zero_division=0, output_dict=True,
    )

    print(f"=== {model_name} ===")
    print(f"Macro-F1: {macro_f1:.4f} | Accuracy: {accuracy:.4f}")

    return {
        "model_name": model_name,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
    }
