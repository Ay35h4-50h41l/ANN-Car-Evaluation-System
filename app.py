"""Main application entry point for Car Evaluation ANN project."""

from __future__ import annotations

import os

import numpy as np
from tensorflow.keras.models import load_model

from evaluate import evaluate_model
from train import train_model
from visualization import plot_ann_structure, plot_training_curves


ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "car_ann_model.h5")
SPLIT_PATH = os.path.join(ARTIFACTS_DIR, "data_split.npz")


def run_full_pipeline():
    """Run training, evaluation, and visualizations in sequence."""
    print("\n[1/4] Training model...")
    train_model()

    print("\n[2/4] Evaluating model...")
    evaluate_model()

    print("\n[3/4] Showing ANN structure...")
    plot_ann_structure()

    print("\n[4/4] Showing training curves and forward-pass explanation...")
    plot_training_curves()
    explain_forward_pass()


def show_menu():
    """Display a simple menu-driven terminal interface."""
    print("\n===== Car Evaluation ANN System =====")
    print("1. Train model")
    print("2. Evaluate model")
    print("3. Visualize ANN architecture")
    print("4. Visualize training curves")
    print("5. Run full pipeline")
    print("6. Load saved model summary")
    print("0. Exit")


def load_saved_model_summary():
    """Load and print saved model architecture summary."""
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Please train the model first.")
        return

    model = load_model(MODEL_PATH)
    print("\nLoaded model successfully. Architecture summary:\n")
    model.summary()


def main():
    """Main interactive loop."""
    while True:
        show_menu()
        choice = input("Select an option: ").strip()

        try:
            if choice == "1":
                train_model()
                print("Training finished and artifacts saved.")
            elif choice == "2":
                evaluate_model()
            elif choice == "3":
                plot_ann_structure()
            elif choice == "4":
                plot_training_curves()
            elif choice == "5":
                run_full_pipeline()
            elif choice == "6":
                load_saved_model_summary()
            elif choice == "0":
                print("Exiting application.")
                break
            else:
                print("Invalid option. Please select a valid menu item.")
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    # Ensure reproducible NumPy behavior where applicable.
    np.random.seed(42)
    main()
