"""Physics-informed feature engineering for MAGIC Gamma Telescope data."""

import numpy as np
import pandas as pd

RAW_FEATURES = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1",
    "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist",
]

ENGINEERED_FEATURES = [
    "ellipticity",
    "shower_density",
    "miss_parameter",
    "conc_ratio",
    "m3_magnitude",
    "size_conc",
    "long_asymmetry",
]

ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

_EPS = 1e-9


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append 7 physics-informed features to a DataFrame of raw Hillas params."""
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing raw features: {missing}")

    out = df.copy()
    out["ellipticity"] = out["fLength"] / (out["fWidth"] + _EPS)
    out["shower_density"] = out["fSize"] / (out["fLength"] * out["fWidth"] + _EPS)
    out["miss_parameter"] = out["fDist"] * np.sin(np.deg2rad(out["fAlpha"]))
    out["conc_ratio"] = out["fConc"] / (out["fConc1"] + _EPS)
    out["m3_magnitude"] = np.sqrt(out["fM3Long"] ** 2 + out["fM3Trans"] ** 2)
    out["size_conc"] = out["fSize"] * out["fConc"]
    out["long_asymmetry"] = out["fAsym"] / (out["fLength"] + _EPS)
    return out


def validate_physical_constraints(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask: True for rows that satisfy all physical constraints."""
    mask = (
        (df["fLength"] > 0)
        & (df["fWidth"] > 0)
        & (df["fLength"] >= df["fWidth"])
        & (df["fConc"] >= 0) & (df["fConc"] <= 1)
        & (df["fConc1"] >= 0) & (df["fConc1"] <= 1)
        & (df["fConc"] >= df["fConc1"])
        & (df["fAlpha"] >= 0) & (df["fAlpha"] <= 90)
        & (df["fDist"] > 0)
    )
    return mask
