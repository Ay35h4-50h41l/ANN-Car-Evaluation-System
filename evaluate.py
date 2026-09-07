"""Evaluation module for Car Evaluation ANN project."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model


def evaluate_model(
    artifacts_dir: str = "artifacts",
    model_filename: str = "car_ann_model.h5",
    split_filename: str = "data_split.npz",
    classes_filename: str = "label_classes.npy",
) -> Dict[str, float]:
    """Load trained model and evaluate it on the stored test split."""
    model_path = os.path.join(artifacts_dir, model_filename)
    split_path = os.path.join(artifacts_dir, split_filename)
    classes_path = os.path.join(artifacts_dir, classes_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Processed dataset split not found at: {split_path}")

    # Load model and test data.
    model = load_model(model_path)
    data = np.load(split_path)
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Predict class probabilities and convert to class IDs.
    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    # Compute evaluation metrics.
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    # Build detailed report with class names if available.
    if os.path.exists(classes_path):
        class_names = np.load(classes_path, allow_pickle=True)
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    else:
        report = classification_report(y_test, y_pred, zero_division=0)

    # Remove macro/weighted summary rows from the printed report.
    filtered_report = "\n".join(
        line for line in report.splitlines() if "macro avg" not in line and "weighted avg" not in line
    )

    print("\n=== Evaluation Results ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(filtered_report)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
    }


if __name__ == "__main__":
    evaluate_model()
