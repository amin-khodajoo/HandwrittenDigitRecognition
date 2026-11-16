"""
preprocessing.py
================
This module handles all image preprocessing steps required to transform
a user drawing from a Streamlit canvas into a format compatible with the
trained MNIST K-Nearest Neighbors (KNN) model.

The MNIST Dataset standard:
- Digit color: White strokes (~255 or ~1.0) on a pure Black background (~0).
- Dimensions: 28 x 28 pixels (flattened to a 784-dimensional 1D array).
- Normalization: Pixel intensities scaled to [0.0, 1.0].
- Centering: Digit bounding box is fitted inside a 20x20 box preserving aspect
  ratio, then placed into a 28x28 canvas and aligned by center of mass.
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, shift


def is_canvas_empty(image_array: np.ndarray, threshold: float = 15.0, min_pixels: int = 20) -> bool:
    """
    Check whether the canvas is effectively empty (no significant drawing).

    Parameters:
    -----------
    image_array : np.ndarray
        Grayscale inverted image where background is 0 and strokes are > 0.
    threshold : float
        Pixel intensity threshold to consider a pixel as drawn stroke.
    min_pixels : int
        Minimum number of stroke pixels required to consider the canvas non-empty.

    Returns:
    --------
    bool
        True if the canvas is empty, False otherwise.
    """
    stroke_pixels = np.sum(image_array > threshold)
    return bool(stroke_pixels < min_pixels)


def crop_to_bounding_box(image_array: np.ndarray, threshold: float = 15.0) -> Optional[np.ndarray]:
    """
    Find the bounding box around drawn strokes and crop out unnecessary background.

    Parameters:
    -----------
    image_array : np.ndarray
        2D grayscale array where digit is white (>0) and background is black (0).
    threshold : float
        Intensity threshold to separate digit strokes from background noise.

    Returns:
    --------
    Optional[np.ndarray]
        Cropped 2D numpy array containing only the digit bounding box,
        or None if no digit pixels were found.
    """
    rows = np.any(image_array > threshold, axis=1)
    cols = np.any(image_array > threshold, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop with 1px margin safely
    return image_array[rmin : rmax + 1, cmin : cmax + 1]


def resize_and_center_digit(
    cropped_digit: np.ndarray,
    target_box_size: int = 20,
    final_canvas_size: int = 28
) -> np.ndarray:
    """
    Preserve aspect ratio, fit digit inside a target box (default 20x20),
    and center it inside a final square canvas (default 28x28), matching MNIST standards.

    Parameters:
    -----------
    cropped_digit : np.ndarray
        2D array of the cropped digit.
    target_box_size : int
        The maximum width/height of the digit inside the 28x28 frame (MNIST uses 20).
    final_canvas_size : int
        The size of the output square image (MNIST uses 28).

    Returns:
    --------
    np.ndarray
        28x28 grayscale image with centered digit.
    """
    h, w = cropped_digit.shape

    # Calculate scale factor preserving aspect ratio
    if h > w:
        new_h = target_box_size
        new_w = max(1, int(round((w * target_box_size) / h)))
    else:
        new_w = target_box_size
        new_h = max(1, int(round((h * target_box_size) / w)))

    # Resize using PIL with high-quality resampling (LANCZOS)
    pil_cropped = Image.fromarray(cropped_digit.astype(np.uint8))
    pil_resized = pil_cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_digit = np.array(pil_resized, dtype=np.float32)

    # Create empty 28x28 canvas (black background = 0)
    canvas = np.zeros((final_canvas_size, final_canvas_size), dtype=np.float32)

    # Place digit at the geometric center initially
    start_y = (final_canvas_size - new_h) // 2
    start_x = (final_canvas_size - new_w) // 2
    canvas[start_y : start_y + new_h, start_x : start_x + new_w] = resized_digit

    # Shift canvas to align center of mass with the center of the image (13.5, 13.5),
    # which is the exact convention used to create the original MNIST dataset.
    cy, cx = center_of_mass(canvas)
    if not (np.isnan(cy) or np.isnan(cx)):
        target_center = (final_canvas_size - 1) / 2.0  # 13.5 for 28x28
        shift_y = np.clip(target_center - cy, -3.0, 3.0)
        shift_x = np.clip(target_center - cx, -3.0, 3.0)
        canvas = shift(canvas, shift=(shift_y, shift_x), mode='constant', cval=0.0)

    # Clip values to valid range [0, 255]
    return np.clip(canvas, 0.0, 255.0)


def preprocess_drawing(
    canvas_image: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    Full preprocessing pipeline for user canvas drawings:
    1. Extract drawing content and convert to grayscale.
    2. Invert colors so digit is white on black background (matching MNIST).
    3. Check for empty canvas.
    4. Remove unnecessary background by cropping to bounding box.
    5. Preserve aspect ratio and resize digit to fit 20x20 area.
    6. Center inside 28x28 image and align center of mass.
    7. Normalize pixel values to [0.0, 1.0].
    8. Reshape to 784-dimensional vector.

    Parameters:
    -----------
    canvas_image : np.ndarray
        RGBA or RGB array returned by streamlit-drawable-canvas.

    Returns:
    --------
    Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]
        - flat_vector: (1, 784) float32 vector ready for model prediction, or None if invalid.
        - digit_28x28: (28, 28) float32 array in [0.0, 1.0] for display preview, or None.
        - error_message: String describing error/empty status, or None if successful.
    """
    if canvas_image is None or canvas_image.size == 0:
        return None, None, "Canvas is empty. Please draw a digit first."

    # Convert to PIL Image for reliable color channel handling
    if isinstance(canvas_image, np.ndarray):
        # Canvas typically produces (H, W, 4) RGBA
        if canvas_image.ndim == 3 and canvas_image.shape[2] == 4:
            pil_img = Image.fromarray(canvas_image.astype(np.uint8), mode='RGBA')
        elif canvas_image.ndim == 3 and canvas_image.shape[2] == 3:
            pil_img = Image.fromarray(canvas_image.astype(np.uint8), mode='RGB')
        else:
            pil_img = Image.fromarray(canvas_image.astype(np.uint8))
    else:
        pil_img = canvas_image

    # Convert to Grayscale ('L' mode: 0-255)
    gray_pil = pil_img.convert('L')
    gray_arr = np.array(gray_pil, dtype=np.float32)

    # Invert colors:
    # Canvas has white background (255) and black stroke (0).
    # MNIST requires black background (0) and white stroke (255).
    # If the user drew on a white canvas: mean pixel value > 128
    if np.mean(gray_arr) > 128:
        inverted = 255.0 - gray_arr
    else:
        # Already dark background
        inverted = gray_arr

    # Remove any faint background noise
    inverted[inverted < 15.0] = 0.0

    # Step 3: Check if canvas is empty
    if is_canvas_empty(inverted):
        return None, None, "Please draw a digit (0-9) on the canvas before predicting."

    # Step 4: Crop to bounding box
    cropped = crop_to_bounding_box(inverted)
    if cropped is None:
        return None, None, "Could not detect a drawn digit. Please try again."

    # Steps 5 & 6: Preserve aspect ratio, resize and center inside 28x28
    digit_28x28 = resize_and_center_digit(cropped, target_box_size=20, final_canvas_size=28)

    # Step 7: Normalize pixel values to 0.0 - 1.0
    normalized_28x28 = digit_28x28 / 255.0

    # Step 8: Flatten into a 784-dimensional vector (1, 784)
    flat_vector = normalized_28x28.reshape(1, 784).astype(np.float32)

    return flat_vector, normalized_28x28, None


def normalize_mnist_data(X: np.ndarray) -> np.ndarray:
    """
    Standardize raw MNIST training/testing samples:
    Converts pixel intensities from [0, 255] to [0.0, 1.0].

    Parameters:
    -----------
    X : np.ndarray
        Array of shape (N, 784) with values in range [0, 255].

    Returns:
    --------
    np.ndarray
        Array of shape (N, 784) with values in range [0.0, 1.0] as float32.
    """
    return (X / 255.0).astype(np.float32)
