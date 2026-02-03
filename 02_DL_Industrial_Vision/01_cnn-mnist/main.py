import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras as ks
import os
import math
import cv2 as cv


OUTPUT_DIR_VISUALS = "output/visuals"


def load_and_explore_data():
    """Load and explore the MNIST dataset.
    Returns:
        Tuple: Training and testing data and labels.
    """
    # Load Dataset MNIST
    (x_train, y_train), (x_test, y_test) = ks.datasets.mnist.load_data()
    print("Training data shape:", x_train.shape)
    print("Test data shape:", x_test.shape)
    print("Unique labels in training data:", np.unique(y_train))
    print("Number of training examples:", x_train.shape[0])
    print("Number of test examples:", x_test.shape[0])
    print("Data format:", tf.keras.backend.image_data_format())

    return x_train, y_train, x_test, y_test


def visualize_sample_images(x_train, y_train, output_dir=OUTPUT_DIR_VISUALS):
    """Visualize sample images from the dataset.
    Args:
        x_train (numpy.ndarray): Training images
        y_train (numpy.ndarray): Training labels
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1 grid of n images
    n = 16
    col = 4
    row = math.ceil(n / col)  # Calculate number of rows needed (ceiling division)
    fig, axes = plt.subplots(row, col, figsize=(8, 8))
    axes = axes.flatten()  # 2D to 1D array for indexing
    for i in range(n):
        axes[i].imshow(x_train[i], cmap="gray")
        axes[i].set_title(f"Label: {y_train[i]}")
        axes[i].axis("off")
    plt.tight_layout()
    for j in range(n, len(axes)):
        axes[j].axis("off")  # Hide unused subplots
    plt.savefig(os.path.join(output_dir, "sample_images_grid.png"))
    plt.close()

    print(f"Sample images saved to '{output_dir}' directory.")
    print("shape of x_train:", x_train.shape)
    print("x_train data type:", x_train.dtype)
    print("x_train min:", x_train.min())
    print("x_train max:", x_train.max())


def normalize_images(x_train, x_test):
    """Normalize images to float32 between 0 and 1.
    Args:
        x_train (numpy.ndarray): Training images
        x_test (numpy.ndarray): Testing images
    Returns:
        Tuple: Normalized training and testing images.
    """
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    print(
        "x_train & x_test:  data type after normalization:", x_train.dtype, " - ",x_test.dtype
    )
    print("x_train: min & max after normalization:", x_train.min(), " - ", x_train.max())
    print("x_test:  min & max after normalization:", x_test.min(), " - ", x_test.max())

    return x_train, x_test


def add_channel_dimension(x_train, x_test):
    """Add channel dimension to images.
    Args:
        x_train (numpy.ndarray): Training images
        x_test (numpy.ndarray): Testing images
    Returns:
        Tuple: Training and testing images with added channel dimension.
    """
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print(" shape of x_train after adding channel dimension:", x_train.shape)
    print(" shape of x_test after adding channel dimension:", x_test.shape)

    return x_train, x_test


if __name__ == "__main__":
    print(
        "================================================== Load and Explore Data =================================================="
    )
    x_train, y_train, x_test, y_test = load_and_explore_data()

    print(
        "================================================== Visualize sample images =================================================="
    )
    visualize = visualize_sample_images(x_train, y_train)
    print(
        "================================================== Normalize Images =================================================="
    )
    x_train, x_test = normalize_images(x_train, x_test)
    print(
        "================================================== Add Channel Dimension =================================================="
    )
    x_train, x_test = add_channel_dimension(x_train, x_test)
