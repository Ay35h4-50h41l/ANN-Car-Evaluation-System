"""Model definition module for Car Evaluation ANN project."""

from __future__ import annotations

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


def build_ann_model(input_dim: int = 6, num_classes: int = 4, learning_rate: float = 0.001):
    """Build and compile ANN with required architecture.

    Architecture:
    - Input: 6 features
    - Hidden: 4 layers, 7 neurons each, sigmoid activation
    - Output: 4 neurons, softmax activation
    """
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(7, activation="sigmoid"),
            Dense(7, activation="sigmoid"),
            Dense(7, activation="sigmoid"),
            Dense(7, activation="sigmoid"),
            Dense(num_classes, activation="softmax"),
        ]
    )

    # Compile the model with Adam + sparse categorical crossentropy.
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
