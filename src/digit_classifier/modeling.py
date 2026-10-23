"""Small, deterministic digit classification experiments.

Data provenance: scikit-learn's public handwritten digits dataset, 8x8 pixel images.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
CLASS_NAMES = list(range(10))


@dataclass(frozen=True)
class DataSplits:
    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    ids_train: np.ndarray
    ids_validation: np.ndarray
    ids_test: np.ndarray


def load_splits(seed: int = SEED) -> DataSplits:
    ds = load_digits()
    x = ds.data.astype(np.float64)
    y = ds.target.astype(int)
    idx = np.arange(len(y))
    # Important: no test data are used for tuning or model selection.
    x_dev, x_test, y_dev, y_test, id_dev, id_test = train_test_split(
        x, y, idx, test_size=0.20, stratify=y, random_state=seed
    )
    x_train, x_val, y_train, y_val, id_train, id_val = train_test_split(
        x_dev, y_dev, id_dev, test_size=0.25, stratify=y_dev, random_state=seed
    )
    return DataSplits(x_train, x_val, x_test, y_train, y_val, y_test,
                      id_train, id_val, id_test)


def build_candidates(seed: int = SEED) -> dict:
    return {
        'Logistic Regression': make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=1, min_samples_leaf=1
        ),
        'RBF SVM': make_pipeline(
            StandardScaler(), SVC(C=5.0, kernel='rbf', probability=True, random_state=seed)
        ),
    }


def validate_pixels(pixels: list[float]) -> None:
    if len(pixels) != 64:
        raise ValueError('Expected exactly 64 grayscale pixel values (8x8).')
    if any(not np.isfinite(p) or p < 0 or p > 16 for p in pixels):
        raise ValueError('Each pixel must be finite and within 0 to 16.')
