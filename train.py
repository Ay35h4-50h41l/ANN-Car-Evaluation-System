"""Training module for Car Evaluation ANN project."""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from data_preprocessing import prepare_datasets
from model import build_ann_model


def train_model(
    csv_path: str = "car_evaluation",
    artifacts_dir: str = "artifacts",
    model_filename: str = "car_ann_model.h5",
    history_filename: str = "training_history.json",
    epochs: int = 120,
    batch_size: int = 32,
    random_state: int = 42,
) -> Tuple[object, Dict[str, list], np.ndarray, np.ndarray]:
    """Train the ANN model and save model + history."""
    # Prepare datasets and persist preprocessing artifacts.
    X_train, X_test, y_train, y_test = prepare_datasets(
        csv_path=csv_path,
        random_state=random_state,
        artifacts_dir=artifacts_dir,
    )

    # Split training data into train/validation using stratification.
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train,
    )

    # Balance class influence to improve minority-class learning.
    classes = np.unique(y_subtrain)
    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_subtrain,
    )
    class_weights = {int(cls): float(weight) for cls, weight in zip(classes, class_weight_values)}

    # Single-run training using epochs.
    model = build_ann_model(input_dim=X_train.shape[1], num_classes=4)
    history = model.fit(
        X_subtrain,
        y_subtrain,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        shuffle=True,
        verbose=1,
    )

    # Save artifacts for later use.
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, model_filename)
    history_path = os.path.join(artifacts_dir, history_filename)

    model.save(model_path)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    return model, history.history, X_test, y_test


if __name__ == "__main__":
    trained_model, hist, _, _ = train_model()
    print("Training complete.")
    print(f"Final training accuracy: {hist['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {hist['val_accuracy'][-1]:.4f}")
