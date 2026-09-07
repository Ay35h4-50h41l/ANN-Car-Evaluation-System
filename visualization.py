"""Visualization module for Car Evaluation ANN project."""

from __future__ import annotations

import json
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

sns.set_style("whitegrid")


def plot_ann_structure(
    input_neurons: int = 6,
    hidden_layers: Optional[List[int]] = None,
    output_neurons: int = 4,
):
    """Visualize ANN structure: input -> hidden layers -> output."""
    if hidden_layers is None:
        hidden_layers = [7, 7, 7, 7]

    layers = [input_neurons] + hidden_layers + [output_neurons]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title("ANN Architecture Visualization", fontsize=14)
    ax.axis("off")

    layer_x = np.linspace(0.1, 0.9, len(layers))
    neuron_positions = []

    # Draw neurons for each layer.
    for x, n_neurons in zip(layer_x, layers):
        ys = np.linspace(0.1, 0.9, n_neurons)
        neuron_positions.append([(x, y) for y in ys])
        ax.scatter([x] * n_neurons, ys, s=220, color="#4C78A8", edgecolors="black", zorder=3)

    # Draw full connections between consecutive layers.
    for i in range(len(neuron_positions) - 1):
        for x1, y1 in neuron_positions[i]:
            for x2, y2 in neuron_positions[i + 1]:
                ax.plot([x1, x2], [y1, y2], color="gray", alpha=0.25, linewidth=0.8)

    layer_labels = ["Input"] + [f"Hidden {i}" for i in range(1, len(hidden_layers) + 1)] + ["Output"]
    for x, label in zip(layer_x, layer_labels):
        ax.text(x, 0.96, label, ha="center", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_training_curves(history_path: str = "artifacts/training_history.json", smooth_window: int = 7):
    """Plot training/validation loss and accuracy over epochs.

    Raw curves can fluctuate across epochs; a moving-average trend is overlaid
    to show the underlying learning pattern more clearly.
    """
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Training history not found at: {history_path}")

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history["loss"]) + 1)

    def smooth(values):
        arr = np.array(values, dtype=float)
        if smooth_window <= 1 or len(arr) < smooth_window:
            return arr
        kernel = np.ones(smooth_window) / smooth_window
        padded = np.pad(arr, (smooth_window // 2, smooth_window - 1 - smooth_window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve.
    axes[0].plot(epochs, history["loss"], alpha=0.3, label="Train Loss (Raw)")
    axes[0].plot(epochs, smooth(history["loss"]), linewidth=2.5, label="Train Loss (Smoothed)")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], alpha=0.3, label="Val Loss (Raw)")
        axes[0].plot(epochs, smooth(history["val_loss"]), linewidth=2.5, label="Val Loss (Smoothed)")
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy curve.
    axes[1].plot(epochs, history["accuracy"], alpha=0.3, label="Train Accuracy (Raw)")
    axes[1].plot(epochs, smooth(history["accuracy"]), linewidth=2.5, label="Train Accuracy (Smoothed)")
    if "val_accuracy" in history:
        axes[1].plot(epochs, history["val_accuracy"], alpha=0.3, label="Val Accuracy (Raw)")
        axes[1].plot(epochs, smooth(history["val_accuracy"]), linewidth=2.5, label="Val Accuracy (Smoothed)")
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def explain_forward_pass(
    model_path: str = "artifacts/car_ann_model.h5",
    split_path: str = "artifacts/data_split.npz",
):
    """Step-by-step visualization of how one input transforms through the network."""
    if not os.path.exists(model_path) or not os.path.exists(split_path):
        raise FileNotFoundError("Model or data split missing. Train the model first.")

    model = load_model(model_path, compile=False)
    data = np.load(split_path)
    sample = data["X_test"][0:1]

    # Ensure the loaded model is built before extracting symbolic inputs/outputs.
    _ = model.predict(sample, verbose=0)

    # Build intermediate model to capture outputs from each Dense layer.
    dense_layers = [layer.output for layer in model.layers]
    intermediate = Model(inputs=model.inputs, outputs=dense_layers)
    outputs = intermediate.predict(sample, verbose=0)

    step_messages = [
        "Input entering network",
        "Weights being applied",
        "Activation function applied",
        "Output generated",
    ]

    fig, ax = plt.subplots(figsize=(10, 4))

    def update(frame):
        ax.clear()
        values = outputs[frame].flatten()
        ax.bar(range(len(values)), values, color="#59A14F")
        ax.set_ylim(0, 1)
        ax.set_title(f"Layer {frame + 1} activation values")
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Activation")

        # Rotate explanatory statements across animation frames.
        message = step_messages[min(frame, len(step_messages) - 1)]
        ax.text(0.02, 0.92, message, transform=ax.transAxes, fontsize=11, fontweight="bold")

    _ = FuncAnimation(fig, update, frames=len(outputs), interval=1000, repeat=False)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
):
    """Optional extra visualization: confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
