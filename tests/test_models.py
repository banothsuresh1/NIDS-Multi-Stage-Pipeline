import numpy as np

from config import CONV1D_FILTERS, CONV1D_KERNEL_SIZE
from config import K as K_DEFAULT
from config import LSTM_DROPOUT, LSTM_EMBED_DIM, LSTM_UNITS, MAX_SEQ_LEN, VOCAB, VOCAB_SIZE
from src.nids.models import build_conv1d, build_lstm, build_sequence_model, evaluate_classifier, train_lstm


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
    model = build_lstm(VOCAB_SIZE, LSTM_EMBED_DIM, LSTM_UNITS, LSTM_DROPOUT, n_classes)
    assert model.input_shape == (None, MAX_SEQ_LEN)
    assert model.output_shape == (None, n_classes)
    # Embedding's input_dim must cover VOCAB_SIZE = len(VOCAB) + 1 (id 0
    # is reserved for padding; real tokens are ids 1..len(VOCAB)).
    embedding_layer = model.layers[0]
    assert embedding_layer.input_dim == VOCAB_SIZE


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


def test_build_conv1d_input_output_shape(preprocessed_data):
    n_classes = preprocessed_data["K"]
    model = build_conv1d(VOCAB_SIZE, LSTM_EMBED_DIM, CONV1D_FILTERS, CONV1D_KERNEL_SIZE,
                          LSTM_DROPOUT, n_classes)
    assert model.input_shape == (None, MAX_SEQ_LEN)
    assert model.output_shape == (None, n_classes)


def test_build_sequence_model_dispatches_on_model_type(preprocessed_data):
    n_classes = preprocessed_data["K"]
    bilstm = build_sequence_model(VOCAB_SIZE, LSTM_EMBED_DIM, LSTM_UNITS, LSTM_DROPOUT,
                                   n_classes, model_type="bilstm")
    conv1d = build_sequence_model(VOCAB_SIZE, LSTM_EMBED_DIM, LSTM_UNITS, LSTM_DROPOUT,
                                   n_classes, model_type="conv1d")
    assert any("bidirectional" in layer.name for layer in bilstm.layers)
    assert any("conv1d" in layer.name for layer in conv1d.layers)


def test_train_lstm_runs_one_epoch_via_tf_data():
    # Smoke test for the tf.data-based training path (cache/shuffle/batch/
    # prefetch instead of raw numpy arrays fed to model.fit) -- catches
    # tuple-unpacking or dtype mistakes without paying for a real epoch.
    rng = np.random.default_rng(0)
    n, seq_len, n_classes = 40, MAX_SEQ_LEN, 3
    X = rng.integers(0, VOCAB_SIZE, size=(n, seq_len))
    y = rng.integers(0, n_classes, size=n)
    sample_weights = np.ones(n, dtype=np.float32)

    model = build_lstm(VOCAB_SIZE, 4, 4, 0.1, n_classes)
    history = train_lstm(
        model, X[:30], y[:30], sample_weights[:30], X[30:], y[30:],
        epochs=1, batch_size=8, patience=5,
    )
    assert "loss" in history.history
    assert len(history.history["loss"]) == 1
