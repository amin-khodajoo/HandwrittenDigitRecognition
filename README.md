# Handwritten Digit Recognition using KNN and MNIST

An interactive, beginner-friendly Machine Learning web application built with **Python**, **Scikit-learn**, and **Streamlit**. Users can draw handwritten digits (0–9) on a digital canvas using their mouse or touchscreen, and a trained **K-Nearest Neighbors (KNN)** model predicts the digit in real time.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [Machine Learning Details](#machine-learning-details)
4. [Image Preprocessing Pipeline](#image-preprocessing-pipeline)
5. [Model Evaluation & Results](#model-evaluation--results)
6. [Project Structure](#project-structure)
7. [Local Setup & Installation](#local-setup--installation)
8. [Training the Model](#training-the-model)
9. [Running the Application](#running-the-application)
10. [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
11. [How the Application Works](#how-the-application-works)

---

## 1. Project Overview

The goal of this project is to recognize handwritten digits without relying on complex deep learning frameworks (such as TensorFlow or PyTorch). Instead, it demonstrates the power and interpretability of classical Machine Learning using the **K-Nearest Neighbors (KNN)** algorithm trained on the benchmark **MNIST** dataset.

### Key Highlights:
- **Interactive Canvas:** Draw digits directly in your web browser.
- **Robust Preprocessing:** Automatically centers, scales, and aligns drawn digits to match the exact distribution of MNIST images.
- **Instant Predictions:** Predicts the digit and visualizes neighbor voting consensus.
- **Explainable Confidence:** Transparent breakdown of top-3 nearest neighbor distances and class voting distribution.
- **Free Cloud Deployment:** Ready to deploy with one click to Streamlit Community Cloud.

---

## 2. Technology Stack

- **Python 3.10+**: Core programming language.
- **Scikit-learn**: KNN classification algorithm (`KNeighborsClassifier`), dataset fetcher, and evaluation metrics.
- **NumPy**: Matrix operations, pixel scaling, and array reshaping.
- **SciPy**: Center-of-mass calculation and translation shifts.
- **Pillow (PIL)**: Image cropping, aspect-ratio-preserving resampling, and color conversions.
- **Matplotlib**: Confusion matrix plotting and visualization export.
- **Streamlit**: Web framework for interactive browser UI.
- **Streamlit Drawable Canvas**: HTML5/Fabric.js canvas for drawing digits.
- **Joblib**: Fast model serialization and compression.

---

## 3. Machine Learning Details

### Dataset Configuration
- **Dataset:** MNIST (Modified National Institute of Standards and Technology)
- **Training Samples:** 10,000 samples
- **Testing Samples:** 10,000 samples
- **Image Dimensions:** 28 × 28 pixels = 784 numerical features
- **Classes:** 10 classes (digits 0 through 9)

### Algorithm Configuration
- **Model:** K-Nearest Neighbors (`KNeighborsClassifier`)
- **Number of Neighbors ($k$):** 3
- **Distance Metric:** Euclidean distance ($L_2$ norm)
- **Voting Scheme:** Uniform majority voting among the 3 nearest neighbors

### Why KNN with 10,000 Samples?
KNN stores all training vectors in memory and computes Euclidean distances at inference time. Using 10,000 samples provides an optimal balance:
- High accuracy (**~94.6%** on test data).
- Ultra-low inference latency (**~0.16 ms** per sample).
- Compact serialized model file (**~2.65 MB**), easily within Streamlit Community Cloud free tier memory limits.

---

## 4. Image Preprocessing Pipeline

User drawings and MNIST images have very different characteristics:
- Canvas produces **black strokes on a white background**.
- MNIST contains **white digits on a black background**, centered inside a 28×28 grid.

To bridge this gap, `preprocessing.py` performs an 8-step pipeline:

```
[User Canvas (280x280 RGBA)]
           │
           ▼
1. Convert to Grayscale
           │
           ▼
2. Color Inversion (White stroke on Black background)
           │
           ▼
3. Empty Canvas Check (Gracefully handles blank drawings)
           │
           ▼
4. Crop to Bounding Box (Removes unnecessary margins)
           │
           ▼
5. Preserve Aspect Ratio & Scale into 20x20 Box
           │
           ▼
6. Center in 28x28 Canvas & Align Center-of-Mass
           │
           ▼
7. Normalize Intensities to [0.0, 1.0]
           │
           ▼
8. Flatten to 784-Dimensional Feature Vector (1, 784)
```

---

## 5. Model Evaluation & Results

The model was evaluated on all **10,000 independent test samples** from the MNIST benchmark.

### Performance Summary:
| Metric | Value |
|---|---|
| **Accuracy** | **94.63%** |
| **Precision (Macro Avg)** | **0.9474** |
| **Recall (Macro Avg)** | **0.9457** |
| **F1-Score (Macro Avg)** | **0.9460** |
| **Training Time** | **0.10 seconds** |
| **Test Set Prediction Time** | **1.56 seconds** (~0.16 ms/sample) |

### Per-Class Performance:
| Digit | Precision | Recall | F1-Score | Support |
|:---:|:---:|:---:|:---:|:---:|
| **0** | 0.9436 | 0.9898 | 0.9661 | 980 |
| **1** | 0.9254 | 0.9947 | 0.9588 | 1,135 |
| **2** | 0.9705 | 0.9254 | 0.9474 | 1,032 |
| **3** | 0.9323 | 0.9406 | 0.9364 | 1,010 |
| **4** | 0.9603 | 0.9348 | 0.9474 | 982 |
| **5** | 0.9490 | 0.9383 | 0.9436 | 892 |
| **6** | 0.9622 | 0.9833 | 0.9726 | 958 |
| **7** | 0.9379 | 0.9407 | 0.9393 | 1,028 |
| **8** | 0.9715 | 0.8758 | 0.9212 | 974 |
| **9** | 0.9208 | 0.9336 | 0.9272 | 1,009 |

The full report and confusion matrix plot are saved under:
- `results/confusion_matrix.png`
- `results/evaluation_report.txt`

---

## 6. Project Structure

```
HandwrittenDigitRecognition/
│
├── app.py                     # Streamlit user interface & interactive canvas
├── train.py                   # Script to train KNN model & generate evaluation metrics
├── predict.py                 # Prediction utility for loading model & inference
├── preprocessing.py           # Image preprocessing pipeline (canvas -> MNIST format)
├── requirements.txt           # Python dependencies for local & cloud deployment
├── README.md                  # Comprehensive documentation & setup instructions
│
├── model/
│   └── knn_model.pkl          # Trained & compressed KNN model (Joblib)
│
└── results/
    ├── confusion_matrix.png   # Matplotlib confusion matrix plot
    └── evaluation_report.txt  # Detailed evaluation metrics & per-class scores
```

---

## 7. Local Setup & Installation

### Step 1: Clone or Open the Repository
```bash
git clone https://github.com/<your-username>/HandwrittenDigitRecognition.git
cd HandwrittenDigitRecognition
```

### Step 2: Create and Activate a Virtual Environment
**On Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 8. Training the Model

The pre-trained model `model/knn_model.pkl` is already included. If you want to retrain the model or regenerate evaluation results:

```bash
python train.py
```

This will:
1. Download MNIST data (cached locally).
2. Train the KNN ($k=3$) model on 10,000 samples.
3. Evaluate on 10,000 test samples.
4. Save the trained model to `model/knn_model.pkl`.
5. Save `results/confusion_matrix.png` and `results/evaluation_report.txt`.

---

## 9. Running the Application

To launch the web interface locally, run:

```bash
streamlit run app.py
```

Streamlit will start a local web server and automatically open the application in your browser (usually at `http://localhost:8501`).

---

## 10. Deploying to Streamlit Community Cloud

The project is structured to deploy for free on [Streamlit Community Cloud](https://streamlit.io/cloud) in just a few minutes.

### Step 1: Push Code to GitHub
1. Create a new repository on [GitHub](https://github.com) named `HandwrittenDigitRecognition`.
2. Initialize and push your code:
```bash
git init
git add .
git commit -m "Initial commit: Handwritten Digit Recognition with KNN"
git branch -M main
git remote add origin https://github.com/<your-username>/HandwrittenDigitRecognition.git
git push -u origin main
```
> **Note:** Ensure `model/knn_model.pkl` is committed and pushed. At 2.65 MB, it is well below GitHub's 100 MB file limit and allows instant cloud startup without retraining.

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
2. Click **"New app"** (or **"Create app"**).
3. Fill in the deployment details:
   - **Repository:** `<your-username>/HandwrittenDigitRecognition`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy!"**.

Streamlit Cloud will read `requirements.txt`, install dependencies, load the pre-trained `knn_model.pkl`, and launch your live web application!

---

## 11. How the Application Works

1. **User Input:**
   The user draws any digit between 0 and 9 onto the white 280×280 canvas with the brush tool.
2. **Preprocessing (`preprocessing.py`):**
   When the user clicks **Predict Digit**, the drawing is converted to grayscale and inverted so that the digit is white against a black background. The bounding box of the stroke is computed and extracted, resized preserving aspect ratio into a 20×20 box, placed into a 28×28 square, aligned by its center-of-mass, and normalized to values between 0.0 and 1.0.
3. **Inference (`predict.py`):**
   The flattened 784-dimensional vector is passed to the cached KNN model. The model computes Euclidean distances to the 10,000 training examples, selects the 3 nearest neighbors, and casts a majority vote.
4. **Display (`app.py`):**
   The predicted digit is displayed in large typography alongside the neighbor voting confidence percentage, a probability distribution bar chart (0–9), side-by-side visual inspection of the preprocessed 28×28 image, and neighbor distance metrics.
