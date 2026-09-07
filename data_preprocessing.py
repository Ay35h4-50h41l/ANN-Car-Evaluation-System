"""Data preprocessing module for Car Evaluation ANN project."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder


# Column names for the Car Evaluation dataset.
COLUMN_NAMES = [
    "buying",
    "maint",
    "doors",
    "persons",
    "lug_boot",
    "safety",
    "class",
]


# Fallback source in case local CSV is not available.
UCI_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"


def _load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the dataset from local file, or download from UCI if missing."""
    candidate_paths = [csv_path, f"{csv_path}.csv", "car_evaluation", "car_evaluation.csv"]

    for path in candidate_paths:
        if os.path.exists(path):
            # Read as headerless first (UCI/local raw format expected).
            data = pd.read_csv(path, header=None)
            if data.shape[1] == 7:
                data.columns = COLUMN_NAMES
                return data

            # Fallback: handle files that already include headers.
            data = pd.read_csv(path)
            if set(COLUMN_NAMES).issubset(data.columns):
                return data

    # Download fallback dataset with explicit column names.
    return pd.read_csv(UCI_DATA_URL, names=COLUMN_NAMES)


def prepare_datasets(
    csv_path: str = "car_evaluation",
    test_size: float = 0.2,
    random_state: int = 42,
    artifacts_dir: str = "artifacts",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess the Car Evaluation dataset and return train/test arrays.

    Steps:
    1) Load dataset
    2) Encode categorical features into 6 numeric inputs
    3) Normalize feature values
    4) Label-encode target classes
    5) Train/test split
    6) Persist preprocessing artifacts
    """
    data = _load_dataset(csv_path)

    # Separate features and target.
    X = data[COLUMN_NAMES[:-1]].copy()
    y = data[COLUMN_NAMES[-1]].copy()

    # Encode categorical input features using one-hot encoding.
    X_clean = X.apply(lambda col: col.astype(str).str.strip().str.lower())
    feature_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_encoded = feature_encoder.fit_transform(X_clean)

    # Normalize feature values to [0, 1].
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Encode target labels into class indices [0..3].
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str).str.strip().str.lower())

    # Split data with stratification for balanced class distribution.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    # Save artifacts for reproducible training/evaluation.
    os.makedirs(artifacts_dir, exist_ok=True)
    np.save(
        os.path.join(artifacts_dir, "feature_categories.npy"),
        np.array(feature_encoder.categories_, dtype=object),
        allow_pickle=True,
    )
    np.save(os.path.join(artifacts_dir, "scaler_min.npy"), scaler.min_)
    np.save(os.path.join(artifacts_dir, "scaler_scale.npy"), scaler.scale_)
    np.save(os.path.join(artifacts_dir, "label_classes.npy"), label_encoder.classes_)
    np.savez(
        os.path.join(artifacts_dir, "data_split.npz"),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    return X_train, X_test, y_train, y_test
