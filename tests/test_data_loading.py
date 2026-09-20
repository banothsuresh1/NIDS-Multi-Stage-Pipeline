import numpy as np

from src.nids.data_loading import load_day, load_split


def test_columns_stripped(synthetic_csv_dir):
    df = load_day(1, synthetic_csv_dir)
    assert all(col == col.strip() for col in df.columns)


def test_label_column_renamed(synthetic_csv_dir):
    df = load_day(1, synthetic_csv_dir)
    assert "Label" in df.columns
    assert " Label" not in df.columns


def test_day_column_added(synthetic_csv_dir):
    df = load_day(1, synthetic_csv_dir)
    assert "Day" in df.columns
    assert (df["Day"] == 1).all()


def test_no_duplicate_header_rows(synthetic_csv_dir):
    df = load_day(1, synthetic_csv_dir)
    assert not (df["Label"] == "Label").any()


def test_shape_positive(synthetic_csv_dir):
    df = load_day(1, synthetic_csv_dir)
    assert df.shape[0] > 0
    assert df.shape[1] > 0


def test_load_split_concatenates_days(synthetic_csv_dir):
    df = load_split([1, 2], synthetic_csv_dir, label="TEST")
    df1 = load_day(1, synthetic_csv_dir)
    df2 = load_day(2, synthetic_csv_dir)
    assert df.shape[0] == df1.shape[0] + df2.shape[0]
    assert set(df["Day"].unique()) == {1, 2}
