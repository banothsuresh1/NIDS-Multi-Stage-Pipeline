import numpy as np

from config import K as K_DEFAULT
from config import LSTM_DROPOUT, LSTM_EMBED_DIM, LSTM_UNITS, MAX_SEQ_LEN, VOCAB
from src.nids.models import build_lstm, evaluate_classifier


def test_train_rf_predict_proba_shape(trained_rf, preprocessed_data):
    d = preprocessed_data
    proba = trained_rf.predict_proba(d["X_test"])
    assert proba.shape[0] == d["X_test"].shape[0]
    assert proba.shape[1] == d["K"]


def test_rf_probabilities_sum_to_one(trained_rf, preprocessed_data):
    proba = trained_rf.predict_proba(preprocessed_data["X_test"])
    sums = proba.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_build_lstm_input_output_shape(preprocessed_data):
    n_classes = preprocessed_data["K"]
    model = build_lstm(len(VOCAB), LSTM_EMBED_DIM, LSTM_UNITS, LSTM_DROPOUT, n_classes)
    assert model.input_shape == (None, MAX_SEQ_LEN)
    assert model.output_shape == (None, n_classes)


def test_train_xgboost_predict_proba_shape(trained_xgb, preprocessed_data):
    d = preprocessed_data
    proba = trained_xgb.predict_proba(d["X_test"])
    assert proba.shape[0] == d["X_test"].shape[0]
    assert proba.shape[1] == d["K"]


def test_evaluate_classifier_keys(trained_rf, preprocessed_data):
    d = preprocessed_data
    metrics = evaluate_classifier(trained_rf, d["X_test"], d["y_test"], "Random Forest", d["classes"])
    assert "macro_f1" in metrics
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
