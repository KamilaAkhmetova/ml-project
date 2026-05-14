import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    ALL_FEATURES,
    ENGINEERED_FEATURES,
    RAW_FEATURES,
    engineer_features,
    validate_physical_constraints,
)


@pytest.fixture
def sample_row() -> pd.DataFrame:
    return pd.DataFrame([{
        "fLength": 30.0, "fWidth": 15.0, "fSize": 2.5, "fConc": 0.4, "fConc1": 0.2,
        "fAsym": 6.0, "fM3Long": 8.0, "fM3Trans": 6.0, "fAlpha": 30.0, "fDist": 50.0,
    }])


def test_engineer_features_adds_all_columns(sample_row):
    out = engineer_features(sample_row)
    assert all(c in out.columns for c in ALL_FEATURES)
    assert len(out) == 1


def test_engineer_features_formulas(sample_row):
    out = engineer_features(sample_row).iloc[0]
    assert out["ellipticity"] == pytest.approx(30.0 / 15.0)
    assert out["shower_density"] == pytest.approx(2.5 / (30.0 * 15.0))
    assert out["miss_parameter"] == pytest.approx(50.0 * np.sin(np.deg2rad(30.0)))
    assert out["conc_ratio"] == pytest.approx(0.4 / 0.2)
    assert out["m3_magnitude"] == pytest.approx(np.sqrt(64.0 + 36.0))
    assert out["size_conc"] == pytest.approx(2.5 * 0.4)
    assert out["long_asymmetry"] == pytest.approx(6.0 / 30.0)


def test_engineer_features_raises_on_missing():
    bad = pd.DataFrame([{"fLength": 1.0}])
    with pytest.raises(ValueError):
        engineer_features(bad)


@pytest.mark.parametrize("col,bad_value", [
    ("fLength", -1.0),
    ("fWidth", -1.0),
    ("fConc", 1.5),
    ("fConc1", -0.1),
    ("fAlpha", 100.0),
    ("fDist", -1.0),
])
def test_validate_rejects_individual_violations(sample_row, col, bad_value):
    row = sample_row.copy()
    row[col] = bad_value
    mask = validate_physical_constraints(row)
    assert not mask.iloc[0]


def test_validate_rejects_fwidth_greater_than_flength(sample_row):
    row = sample_row.copy()
    row["fWidth"] = 100.0
    assert not validate_physical_constraints(row).iloc[0]


def test_validate_rejects_fconc1_greater_than_fconc(sample_row):
    row = sample_row.copy()
    row["fConc"] = 0.1
    row["fConc1"] = 0.5
    assert not validate_physical_constraints(row).iloc[0]


def test_validate_accepts_valid_row(sample_row):
    assert validate_physical_constraints(sample_row).iloc[0]


def test_engineered_features_count():
    assert len(ENGINEERED_FEATURES) == 7
    assert len(RAW_FEATURES) == 10
    assert len(ALL_FEATURES) == 17
