"""Feature loading and the sklearn preprocessing pipeline.

Leakage: every fitted step (imputer, scaler, variance/correlation filters, SelectKBest) lives inside one
Pipeline, so fitting it on X_train (or on the training part of each CV fold) is all that is ever done.
Classes are balanced (150/144/150/150 per class in train), so no SMOTE/oversampling is applied.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

FEATURE_SETS = ("handcrafted", "handcrafted_norm", "resnet", "both")


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop a feature if |corr| with an earlier kept feature exceeds `threshold` (fit on training data only)."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        c = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)))
        keep = []
        for j in range(c.shape[0]):
            if all(c[i, j] <= self.threshold for i in keep):
                keep.append(j)
        self.support_ = np.zeros(c.shape[0], dtype=bool)
        self.support_[keep] = True
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self):
        return self.support_


def build_pipeline(k=30, scaler="robust", corr_threshold=0.95, var_threshold=0.0):
    """imputer(median) -> scaler -> VarianceThreshold -> [CorrelationFilter] -> SelectKBest(f_classif, k).

    k: number of features kept (int or "all"; tune it via `select__k`). If k exceeds the number of features
    left after filtering, sklearn keeps them all. corr_threshold=None disables the correlation filter.
    scaler: "robust" (default; EDA/feature analysis found skewed, outlier-heavy features, see
    scripts/05_feature_analysis.py) or "standard".
    """
    scalers = {"robust": RobustScaler(), "standard": StandardScaler()}
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", scalers[scaler]),
        ("variance", VarianceThreshold(var_threshold)),
        ("corr", CorrelationFilter(corr_threshold) if corr_threshold is not None else "passthrough"),
        ("select", SelectKBest(f_classif, k=k)),
    ])


def load_dataset(kind="handcrafted"):
    """-> X_train, y_train, X_test, y_test for 'handcrafted', 'handcrafted_norm', 'resnet' or 'both' (official split).

    'handcrafted_norm' is the illumination-corrected / channel-standardised variant (same 116 column names);
    'both' = handcrafted + resnet.

    X are DataFrames (columns: handcrafted names, resnet_0..resnet_2047); y are int arrays (label_id 0-3).
    """
    if kind not in FEATURE_SETS:
        raise ValueError(f"kind must be one of {FEATURE_SETS}")
    H = pd.read_csv(config.DATA_DIR / "features_handcrafted.csv")
    ids = pd.read_csv(config.DATA_DIR / "features_resnet_ids.csv")
    assert (H.image_id.values == ids.image_id.values).all(), "feature files are misaligned"
    parts = []
    if kind in ("handcrafted", "both"):
        parts.append(H.drop(columns=["image_id", "label_id", "split"]))
    if kind == "handcrafted_norm":
        HN = pd.read_csv(config.DATA_DIR / "features_handcrafted_norm.csv")
        assert (HN.image_id.values == H.image_id.values).all() and (HN.label_id.values == H.label_id.values).all()
        parts.append(HN.drop(columns=["image_id", "label_id", "split"]))
    if kind in ("resnet", "both"):
        E = np.load(config.DATA_DIR / "features_resnet.npy")
        parts.append(pd.DataFrame(E, columns=[f"resnet_{i}" for i in range(E.shape[1])]))
    X = pd.concat(parts, axis=1)
    tr, te = (H.split == "train").values, (H.split == "test").values
    y = H.label_id.values.astype(int)
    return X[tr].reset_index(drop=True), y[tr], X[te].reset_index(drop=True), y[te]
