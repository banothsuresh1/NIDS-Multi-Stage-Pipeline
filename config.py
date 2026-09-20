"""
Central configuration for the Flow-Aware Temporal Pattern Mining NIDS pipeline.
Every module and the notebook imports its hyperparameters from here so that a
single change propagates consistently across preprocessing, modelling, fusion,
pattern mining, graph construction, risk scoring, streaming evaluation and
explainability.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Global
# ---------------------------------------------------------------------------
DATASET_DIR = Path(r"D:\IDSPROJECT2026\CIC-IDS2017")
SEED = 42
K = 15  # number of classes in CIC-IDS2017

DAY_FILES = {
    1: "Monday-WorkingHours.pcap_ISCX.csv",
    2: "Tuesday-WorkingHours.pcap_ISCX.csv",
    3: "Wednesday-workingHours.pcap_ISCX.csv",
    4: "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    5: "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    6: "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    7: "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}
TRAIN_DAYS = [1, 2]
VAL_DAYS = [3]
TEST_DAYS = [4, 5, 6, 7]

# ---------------------------------------------------------------------------
# Stage 2 - Preprocessing & imbalance handling
# ---------------------------------------------------------------------------
OUTLIER_CLIP_PERCENTILE = 99
SMOTE_K_NEIGHBORS = 5
ENN_N_NEIGHBORS = 3
SMOTE_MIN_SAMPLES = 50  # augment classes below this count

# ---------------------------------------------------------------------------
# Stage 3 - Feature engineering
# ---------------------------------------------------------------------------
GROUP_A_SIZE = 32
GROUP_B_SIZE = 18
GROUP_C_SIZE = 15
GROUP_D_SIZE = 13

# ---------------------------------------------------------------------------
# Stage 4 - Session reconstruction
# ---------------------------------------------------------------------------
SESSION_TIMEOUT = 60  # seconds

# ---------------------------------------------------------------------------
# Stage 5 - Behavioral event encoding
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 20
VOCAB = [
    "CONNECTION_ATTEMPT", "SESSION_ESTABLISHED", "AUTH_FAIL",
    "AUTH_SUCCESS", "SCAN_ACTIVITY", "DATA_TRANSFER",
    "CONNECTION_TERMINATION", "FORCED_TERMINATION",
    "DOS_INDICATOR", "POST_AUTH_ACTIVITY",
]
TOKEN2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOKEN = {i: t for t, i in TOKEN2ID.items()}
AUTH_PORTS = {22, 21, 23, 3389, 445}

# ---------------------------------------------------------------------------
# Stage 6 - Random Forest
# ---------------------------------------------------------------------------
RF_N_ESTIMATORS = 200
RF_MAX_FEATURES = "sqrt"

# Stage 6 - BiLSTM
LSTM_EMBED_DIM = 32
LSTM_UNITS = 128
LSTM_DROPOUT = 0.3
LSTM_EPOCHS = 30
LSTM_BATCH_SIZE = 256
LSTM_LR = 1e-3
LSTM_PATIENCE = 5

# Stage 6 - XGBoost
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LR = 0.1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8

# ---------------------------------------------------------------------------
# Stage 7 - Adaptive evidence fusion
# ---------------------------------------------------------------------------
FUSION_GRID = {
    "w_A": [0.2, 0.3, 0.4, 0.5],
    "w_B": [0.1, 0.2, 0.3],
    "w_C": [0.1, 0.2, 0.3],
    "w_S": [0.05, 0.10, 0.15],
    "w_T": [0.02, 0.05, 0.08],
}
DEFAULT_FUSION_WEIGHTS = (0.35, 0.25, 0.25, 0.10, 0.05)

# ---------------------------------------------------------------------------
# Stage 8 - Pattern mining
# ---------------------------------------------------------------------------
FP_GROWTH_MIN_SUPPORT = 0.02
PREFIXSPAN_MIN_SUPPORT = 0.02
PREFIXSPAN_MAX_GAP = 8

# ---------------------------------------------------------------------------
# Stage 9-10 - Attack-state graph
# ---------------------------------------------------------------------------
EMA_RHO = 0.9
LAMBDA_DECAY = 0.02
NOVEL_EDGE_WEIGHT = 0.05

# ---------------------------------------------------------------------------
# Stage 11 - Risk meta-learner
# ---------------------------------------------------------------------------
RISK_BENIGN_THRESH = 0.35
RISK_SUSPICIOUS_THRESH = 0.75
META_LR_C = 1.0

# ---------------------------------------------------------------------------
# Stage 12 - Streaming evaluation
# ---------------------------------------------------------------------------
STREAM_WINDOW_SIZE = 60
STREAM_STRIDE = 10

# ---------------------------------------------------------------------------
# Stage 13 - Explainability
# ---------------------------------------------------------------------------
SHAP_SAMPLE_SIZE = 500

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
FIGURE_DPI = 120
PALETTE_NAME = "tab20"
