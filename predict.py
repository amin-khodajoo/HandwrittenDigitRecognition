"""
predict.py
==========
This module handles loading the trained K-Nearest Neighbors (KNN) model
and executing predictions on preprocessed digit feature vectors.

It provides functions for:
1. Loading the persisted model from disk (model/knn_model.pkl).
2. Predicting the digit class, voting distribution, and confidence score.
3. Inspecting the nearest neighbors and their Euclidean distances.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import joblib

# Default model path relative to this script
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "model" / "knn_model.pkl"


def load_model(model_path: Optional[Path] = None):
    """
    Load the saved KNN model from disk.

    Parameters:
    -----------
    model_path : Optional[Path]
        Path to the model file. Defaults to 'model/knn_model.pkl'.

    Returns:
    --------
    KNeighborsClassifier
        The loaded scikit-learn KNN model.

    Raises:
    -------
    FileNotFoundError
        If the model file does not exist.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at '{path}'. "
            f"Please run 'python train.py' first to train and save the model."
        )

    model = joblib.load(path)
    return model


def predict_digit(model, flat_vector: np.ndarray) -> Dict[str, Any]:
    """
    Predict the handwritten digit from a 784-dimensional feature vector.

    Parameters:
    -----------
    model : KNeighborsClassifier
        Trained scikit-learn KNN model.
    flat_vector : np.ndarray
        A (1, 784) normalized float32 feature array.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'predicted_digit': int (0-9)
        - 'confidence_pct': float (0-100%)
        - 'probabilities': np.ndarray of shape (10,)
        - 'neighbor_distances': np.ndarray of the k nearest neighbor distances
        - 'neighbor_indices': np.ndarray of the k nearest training sample indices
    """
    if flat_vector.ndim == 1:
        flat_vector = flat_vector.reshape(1, -1)

    if flat_vector.shape[1] != 784:
        raise ValueError(
            f"Expected input with 784 features, but got shape {flat_vector.shape}"
        )

    # 1. Class prediction (majority vote among k neighbors)
    predicted_class = int(model.predict(flat_vector)[0])

    # 2. Probability distribution across all 10 classes (0-9)
    # Scikit-learn's predict_proba calculates the voting fraction:
    # (number of neighbors of class c) / (k total neighbors)
    raw_probs = model.predict_proba(flat_vector)[0]

    # Map to fixed 10-digit array in case any class was absent in classes_
    probabilities = np.zeros(10, dtype=np.float32)
    for idx, cls in enumerate(model.classes_):
        if 0 <= cls < 10:
            probabilities[int(cls)] = raw_probs[idx]

    # 3. Confidence score for the predicted digit (as percentage)
    confidence_pct = float(probabilities[predicted_class] * 100.0)

    # 4. Nearest neighbors distances and indices for explainability
    distances, indices = model.kneighbors(flat_vector)

    return {
        "predicted_digit": predicted_class,
        "confidence_pct": confidence_pct,
        "probabilities": probabilities,
        "neighbor_distances": distances[0],
        "neighbor_indices": indices[0],
    }


def main():
    """
    Test prediction module with a dummy synthetic sample.
    """
    print("Testing predict.py...")
    try:
        model = load_model()
        print(f"Model loaded successfully! Type: {type(model).__name__}")
        print(f"Number of neighbors (k): {model.n_neighbors}")
        print(f"Distance metric: {model.metric}")

        # Test with random normalized vector
        dummy_input = np.zeros((1, 784), dtype=np.float32)
        result = predict_digit(model, dummy_input)
        print(f"Prediction on blank sample:")
        print(f"  Predicted Digit: {result['predicted_digit']}")
        print(f"  Confidence: {result['confidence_pct']:.1f}%")
        print(f"  Probabilities: {result['probabilities']}")
        print(f"  Nearest neighbor distances: {result['neighbor_distances']}")
    except FileNotFoundError as err:
        print(f"Error: {err}")


if __name__ == "__main__":
    main()
