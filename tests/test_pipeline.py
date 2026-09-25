"""Sanity tests: run from the project root with  python -m pytest -q"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocess import (FEATURE_COLS, MARKET_COLS, QUOTE_COLS, ROOT, build_reference,  # noqa: E402
                        filter_target_outliers, load_dataset, preprocess_data)


@pytest.fixture(scope="module")
def ref():
    train = filter_target_outliers(load_dataset("train_test"))
    return build_reference(train, [load_dataset("validation")])


def test_excluded_features_are_not_model_inputs():
    assert not set(QUOTE_COLS + MARKET_COLS) & set(FEATURE_COLS)


def test_outlier_filter_removes_677_rows():
    assert len(filter_target_outliers(load_dataset("train_test"))) == 48_000 - 677


@pytest.mark.parametrize("name", ["train_test", "validation", "december_chart_inputs"])
def test_features_complete_and_physical(ref, name):
    out = preprocess_data(load_dataset(name), ref)
    assert out[FEATURE_COLS].isna().sum().sum() == 0
    assert (out["weight"] > 0).all()
    assert (out["circuity"] >= 1).all()  # a road can't be shorter than a straight line


def test_input_not_modified(ref):
    val = load_dataset("validation")
    before = val.copy()
    preprocess_data(val, ref)
    pd.testing.assert_frame_equal(val, before)


@pytest.mark.skipif(not (ROOT / "validation_predictions.csv").is_file(), reason="run src/predict.py first")
def test_submission_format():
    sub = pd.read_csv(ROOT / "validation_predictions.csv")
    assert list(sub.columns) == ["load_id", "predicted_rate"]
    assert len(sub) == 12_000 and sub["load_id"].is_unique
    assert (sub["predicted_rate"] > 0).all()