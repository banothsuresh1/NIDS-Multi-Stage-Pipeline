# Flow-Aware Temporal Pattern Mining for Multi-Stage Network Intrusion Detection (NIDS)

M.Tech Major Project — CIC-IDS2017 Dataset

A 13-stage NIDS pipeline that fuses flow-level tabular classifiers (Random
Forest, XGBoost), session-level sequence modelling (BiLSTM over behavioral
event tokens), unsupervised frequent-pattern mining (FP-Growth +
PrefixSpan), a learned attack-state transition graph with temporal
consistency scoring, an adaptive evidence-fusion + risk meta-learning
layer, streaming (windowed) evaluation, and SHAP/graph-based
explainability.

## Project Structure

```
.
├── NIDS_Pipeline_Complete.ipynb   # Main research notebook (13 stages)
├── requirements.txt               # Pip dependencies
├── config.py                      # All hyperparameters in one place
├── src/nids/                      # Pipeline implementation
│   ├── data_loading.py            # Stage 1  — CIC-IDS2017 loading + chronological split
│   ├── preprocessing.py           # Stage 2  — cleaning, scaling, class weights, SMOTE+ENN
│   ├── feature_engineering.py     # Stage 3  — MI ranking + feature-group assignment
│   ├── sessions.py                # Stage 4  — bidirectional session reconstruction
│   ├── events.py                  # Stage 5  — behavioral event token encoding
│   ├── models.py                  # Stage 6  — Random Forest, BiLSTM, XGBoost
│   ├── fusion.py                  # Stage 7  — adaptive evidence fusion
│   ├── pattern_mining.py          # Stage 8  — FP-Growth + PrefixSpan
│   ├── attack_graph.py            # Stages 9-10 — attack-state graph + temporal consistency
│   ├── risk.py                    # Stage 11 — adaptive risk meta-learner
│   ├── streaming.py               # Stage 12 — streaming (windowed) evaluation
│   └── explainability.py          # Stage 13 — SHAP + graph evidence reporting
└── tests/                         # pytest suite (synthetic mini-dataset, no real data needed)
```

## Dataset

Download **CIC-IDS2017** (Canadian Institute for Cybersecurity) and place
the 7 day CSVs in a single folder:

| Day | File |
|---|---|
| Monday | `Monday-WorkingHours.pcap_ISCX.csv` |
| Tuesday | `Tuesday-WorkingHours.pcap_ISCX.csv` |
| Wednesday | `Wednesday-workingHours.pcap_ISCX.csv` |
| Thursday AM | `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` |
| Thursday PM | `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` |
| Friday AM | `Friday-WorkingHours-Morning.pcap_ISCX.csv` |
| Friday PM | `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` |

Update `DATASET_DIR` in `config.py` to point at that folder (defaults to
`D:\IDSPROJECT2026\CIC-IDS2017`).

The pipeline uses a **chronological split** (no shuffling, to avoid
temporal leakage): Train = Days 1–2, Validation = Day 3, Test = Days 4–7.
All 15 class labels are retained, including extremely rare ones
(Heartbleed N≈11, Infiltration N≈36).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Pipeline

```bash
jupyter notebook NIDS_Pipeline_Complete.ipynb
```

Run all cells top to bottom. Each of the 13 stages imports its logic from
`src/nids/*.py` — the notebook is the orchestration/experiment-tracking
layer; the algorithms live in the package so they're independently
testable. Figures are written to `figures/` as the notebook runs.

## Running the Tests

The test suite uses a synthetic mini-dataset (in `tests/conftest.py`) that
mimics CIC-IDS2017's structure (leading-space columns, the `" Label"`
quirk, duplicate embedded header rows, inf values, a 5-class imbalanced
label distribution), so it runs in seconds without the real ~50 GB dataset:

```bash
pytest tests/ -v
```

## Configuration

All hyperparameters (SMOTE thresholds, feature-group sizes, session
timeout, model hyperparameters, fusion grid, pattern-mining support
thresholds, graph EMA/decay rates, risk thresholds, streaming window/stride,
SHAP sample size) live in `config.py` and are imported by every module and
the notebook, so a single change propagates consistently through the
whole pipeline.

## Known CIC-IDS2017 Data Quirks (handled in `data_loading.py`)

- Column names carry leading whitespace → stripped on load.
- The label column is literally `" Label"` → renamed to `"Label"`.
- Some CSVs contain duplicate header rows embedded as data → dropped.
- `inf` values appear in flow-rate features → replaced with `NaN`, then
  dropped during preprocessing.
- Timestamp format varies across day files → parsed defensively with
  `pd.to_datetime(errors="coerce")`.
