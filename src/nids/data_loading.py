"""
Stage 1 — Data Acquisition & Chronological Partition.

CIC-IDS2017 quirks handled here:
  * column names carry leading/trailing whitespace -> strip on load
  * the label column is literally " Label" -> renamed to "Label"
  * some CSVs contain duplicate header rows embedded as data -> dropped
  * inf values appear in flow-rate features -> left to preprocessing stage
  * timestamp format is inconsistent across days -> parsed defensively
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from config import DAY_FILES, TRAIN_DAYS, VAL_DAYS, TEST_DAYS


def load_day(day_num: int, dataset_dir: Path) -> pd.DataFrame:
    """Load a single CIC-IDS2017 day CSV, clean column names/labels, tag Day."""
    dataset_dir = Path(dataset_dir)
    filename = DAY_FILES[day_num]
    filepath = dataset_dir / filename

    df = pd.read_csv(filepath, low_memory=False, encoding="latin1")

    # Strip whitespace from column names (CIC-IDS2017 has leading spaces).
    df.columns = df.columns.str.strip()

    # The label column is " Label" pre-strip; after strip it is "Label".
    if "Label" not in df.columns:
        for col in df.columns:
            if col.strip().lower() == "label":
                df = df.rename(columns={col: "Label"})
                break

    # Drop duplicate header rows re-appended as data (Label == "Label").
    df = df[df["Label"] != "Label"].reset_index(drop=True)

    # Best-effort timestamp parsing; column name varies by day/tool version.
    for ts_col in ("Timestamp", " Timestamp", "timestamp"):
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            break

    df["Day"] = day_num
    return df


def load_split(day_list: Iterable[int], dataset_dir: Path, label: str = "") -> pd.DataFrame:
    """Load and concatenate a list of days into a single split DataFrame."""
    frames = []
    for day_num in day_list:
        print(f"[{label}] Loading Day {day_num}: {DAY_FILES[day_num]} ...")
        df_day = load_day(day_num, dataset_dir)
        print(f"[{label}] Day {day_num}: {df_day.shape[0]:,} rows, {df_day.shape[1]} cols")
        frames.append(df_day)
    df_split = pd.concat(frames, axis=0, ignore_index=True)
    print(f"[{label}] TOTAL: {df_split.shape[0]:,} rows")
    return df_split


def load_all_splits(dataset_dir: Path):
    """Chronological split: Train = Days 1-2, Val = Day 3, Test = Days 4-7."""
    df_train = load_split(TRAIN_DAYS, dataset_dir, label="TRAIN")
    df_val = load_split(VAL_DAYS, dataset_dir, label="VAL")
    df_test = load_split(TEST_DAYS, dataset_dir, label="TEST")
    return df_train, df_val, df_test


def load_pooled(dataset_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate ALL SEVEN day files into one pooled DataFrame,
    for a stratified (not chronological) split downstream. The "Day"
    column is retained for reference/audit only -- it plays no role in
    determining train/val/test membership once pooled.
    """
    return load_split(sorted(DAY_FILES), dataset_dir, label="POOLED")
