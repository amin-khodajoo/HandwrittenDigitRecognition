"""
train.py
========
This script trains a K-Nearest Neighbors (KNN) classifier on the MNIST dataset,
evaluates its performance on an independent test set, and saves the trained model
and evaluation artifacts (confusion matrix and report).

Key Configuration:
- Training set: 10,000 samples
- Test set: 10,000 samples
- Classifier: KNeighborsClassifier(n_neighbors=3, metric='euclidean', n_jobs=-1)
- Preprocessing: Normalized pixel intensities to [0.0, 1.0], 784-dim vector
- Saved model path: model/knn_model.pkl
- Artifacts: results/confusion_matrix.png, results/evaluation_report.txt
"""

import os
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend suitable for headless scripts
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import joblib

from preprocessing import normalize_mnist_data

# Define base paths relative to this script for portability
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = MODEL_DIR / "knn_model.pkl"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
REPORT_PATH = RESULTS_DIR / "evaluation_report.txt"

# Dataset configuration
TRAIN_SAMPLES = 10000
TEST_SAMPLES = 10000
N_NEIGHBORS = 3
DISTANCE_METRIC = "euclidean"


def load_mnist_data():
    """
    Load MNIST dataset using scikit-learn's fetch_openml.
    Takes 10,000 training samples and 10,000 test samples.

    Returns:
    --------
    Tuple of (X_train, y_train, X_test, y_test)
    """
    print("=" * 60)
    print("1. Loading MNIST Dataset...")
    print("=" * 60)

    # Scikit-learn caches MNIST locally after the first download
    # We use parser='liac-arff' for broad compatibility across Python and pandas versions
    try:
        X, y = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False,
            return_X_y=True,
            parser="liac-arff",
        )
    except Exception as e:
        print(f"liac-arff parser failed ({e}), falling back to auto parser...")
        X, y = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False,
            return_X_y=True,
            parser="auto",
        )

    # Ensure labels are integers 0 to 9
    y = y.astype(np.int64)

    print(f"Total dataset loaded: {X.shape[0]} samples, {X.shape[1]} features per sample.")

    # MNIST standard split has 60,000 training and 10,000 test images.
    # To satisfy project requirements:
    # - 10,000 training samples (first 10,000 from training partition)
    # - 10,000 test samples (samples 60,000 to 70,000 from standard test partition)
    X_train_raw = X[:TRAIN_SAMPLES]
    y_train = y[:TRAIN_SAMPLES]

    X_test_raw = X[60000 : 60000 + TEST_SAMPLES]
    y_test = y[60000 : 60000 + TEST_SAMPLES]

    print(f"Training samples: {len(X_train_raw):,}")
    print(f"Testing samples:  {len(X_test_raw):,}")

    # Step: Preprocessing - Normalize pixel values to [0.0, 1.0]
    print("\n2. Preprocessing Data...")
    print("   - Normalizing pixel values to [0.0, 1.0] range")
    print("   - Flattening each 28x28 image into 784-dimensional vector")
    X_train = normalize_mnist_data(X_train_raw)
    X_test = normalize_mnist_data(X_test_raw)

    return X_train, y_train, X_test, y_test


def train_knn_model(X_train: np.ndarray, y_train: np.ndarray) -> tuple[KNeighborsClassifier, float]:
    """
    Train a K-Nearest Neighbors classifier on the preprocessed training set.

    Parameters:
    -----------
    X_train : np.ndarray
        Normalized training features of shape (N, 784).
    y_train : np.ndarray
        Training labels of shape (N,).

    Returns:
    --------
    tuple[KNeighborsClassifier, float]
        Trained model instance and training elapsed time in seconds.
    """
    print("\n" + "=" * 60)
    print(f"3. Training KNN Model (k={N_NEIGHBORS}, metric='{DISTANCE_METRIC}')...")
    print("=" * 60)

    model = KNeighborsClassifier(
        n_neighbors=N_NEIGHBORS,
        metric=DISTANCE_METRIC,
        weights="uniform",
        algorithm="auto",
        n_jobs=-1,  # Use all available CPU cores for fast parallel distance calculations
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"Training completed in {training_time:.2f} seconds.")
    return model, training_time


def evaluate_model(
    model: KNeighborsClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    training_time: float,
) -> dict:
    """
    Evaluate the trained model on test data, calculate performance metrics,
    and generate evaluation reports and confusion matrix visualization.

    Parameters:
    -----------
    model : KNeighborsClassifier
        Trained KNN model.
    X_test : np.ndarray
        Normalized test features.
    y_test : np.ndarray
        Ground truth test labels.
    training_time : float
        Elapsed time during model training.

    Returns:
    --------
    dict
        Dictionary containing all evaluation metrics.
    """
    print("\n" + "=" * 60)
    print("4. Evaluating Model on Test Dataset (10,000 samples)...")
    print("=" * 60)

    start_pred = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_pred

    # Compute key classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro")
    precision_weighted = precision_score(y_test, y_pred, average="weighted")
    recall_macro = recall_score(y_test, y_pred, average="macro")
    recall_weighted = recall_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, digits=4)

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "training_time": training_time,
        "prediction_time": prediction_time,
        "confusion_matrix": cm,
        "classification_report": clf_report,
    }

    # Print summary to console
    print(f"\n--- Model Performance Summary ---")
    print(f"Accuracy:           {accuracy * 100:.2f}%")
    print(f"Precision (Macro):  {precision_macro:.4f}")
    print(f"Recall (Macro):     {recall_macro:.4f}")
    print(f"F1-Score (Macro):   {f1_macro:.4f}")
    print(f"Training Time:      {training_time:.2f} s")
    print(f"Prediction Time:    {prediction_time:.2f} s ({prediction_time / len(X_test) * 1000:.2f} ms/sample)")
    print("\nDetailed Classification Report:")
    print(clf_report)

    # Save visual confusion matrix
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix_plot(cm, CONFUSION_MATRIX_PATH)

    # Save text report
    save_evaluation_report(metrics, REPORT_PATH)

    return metrics


def save_confusion_matrix_plot(cm: np.ndarray, file_path: Path):
    """
    Plot and save the confusion matrix using Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title("KNN (k=3) Confusion Matrix on MNIST Test Set (10,000 samples)", fontsize=12, pad=12)
    ax.set_xlabel("Predicted Digit", fontsize=11)
    ax.set_ylabel("True Digit", fontsize=11)
    plt.tight_layout()
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion Matrix saved to: {file_path}")


def save_evaluation_report(metrics: dict, file_path: Path):
    """
    Save performance metrics and classification report to a text file.
    """
    content = f"""================================================================================
HANDWRITTEN DIGIT RECOGNITION - MODEL EVALUATION REPORT
================================================================================
Algorithm:               K-Nearest Neighbors (KNN)
Number of Neighbors (k): {N_NEIGHBORS}
Distance Metric:         {DISTANCE_METRIC}
Training Samples:        {TRAIN_SAMPLES:,}
Testing Samples:         {TEST_SAMPLES:,}
Input Dimensions:        28 x 28 pixels (784 features)
Pixel Normalization:     Range [0.0, 1.0]

--------------------------------------------------------------------------------
OVERALL PERFORMANCE METRICS
--------------------------------------------------------------------------------
Accuracy:                {metrics['accuracy'] * 100:.2f}% ({metrics['accuracy']:.4f})
Precision (Macro Avg):   {metrics['precision_macro']:.4f}
Precision (Weighted):    {metrics['precision_weighted']:.4f}
Recall (Macro Avg):      {metrics['recall_macro']:.4f}
Recall (Weighted):       {metrics['recall_weighted']:.4f}
F1-Score (Macro Avg):    {metrics['f1_macro']:.4f}
F1-Score (Weighted):     {metrics['f1_weighted']:.4f}

--------------------------------------------------------------------------------
COMPUTATIONAL EFFICIENCY
--------------------------------------------------------------------------------
Training Time:           {metrics['training_time']:.2f} seconds
Prediction Time:         {metrics['prediction_time']:.2f} seconds (for {TEST_SAMPLES:,} test samples)
Average Inference Latency: {(metrics['prediction_time'] / TEST_SAMPLES) * 1000:.2f} ms per sample

--------------------------------------------------------------------------------
DETAILED PER-CLASS CLASSIFICATION REPORT
--------------------------------------------------------------------------------
{metrics['classification_report']}

--------------------------------------------------------------------------------
CONFUSION MATRIX
--------------------------------------------------------------------------------
{np.array2string(metrics['confusion_matrix'], separator=', ')}
================================================================================
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Evaluation report saved to: {file_path}")


def save_model(model: KNeighborsClassifier, file_path: Path):
    """
    Save the trained KNN model to disk using joblib.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path, compress=3)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Trained model saved to: {file_path} ({file_size_mb:.2f} MB)")


def main():
    """
    Full training and evaluation pipeline execution.
    """
    print("\nStarting Handwritten Digit Recognition Training Pipeline...")
    X_train, y_train, X_test, y_test = load_mnist_data()
    model, training_time = train_knn_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, training_time)
    save_model(model, MODEL_PATH)
    print("\nPipeline finished successfully! Model is ready for deployment.\n")


if __name__ == "__main__":
    main()
