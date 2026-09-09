"""
mnist_loader.py
---------------
Downloads (once) and loads the MNIST handwritten-digit dataset.

MNIST is stored in a simple binary "IDX" format.  We parse it by hand using
only the Python standard library (urllib, gzip, struct) plus NumPy -- no
special dataset library needed.

The four files are:
    train-images-idx3-ubyte.gz   60000 training images (28x28 pixels)
    train-labels-idx1-ubyte.gz   60000 training labels (0-9)
    t10k-images-idx3-ubyte.gz    10000 test images
    t10k-labels-idx1-ubyte.gz    10000 test labels

Usage
-----
    from mnist_loader import load_mnist
    X_train, y_train, X_test, y_test = load_mnist()
"""

import gzip
import os
import struct
import urllib.request

import numpy as np

# A reliable public mirror of the MNIST files.
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# Store the downloaded files next to this script, in a "data" folder.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _download(filename):
    """Download one MNIST file into DATA_DIR if it is not already there."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        url = BASE_URL + filename
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    return path


def _read_images(path):
    """Read an IDX image file into an array of shape [n_images, 784]."""
    with gzip.open(path, "rb") as f:
        # Header: magic number, count, rows, cols  (4 big-endian int32 values).
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "Not an MNIST image file"
        buf = f.read(rows * cols * num)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # Flatten each 28x28 image into a 784-length row, and scale to [0, 1].
        return data.reshape(num, rows * cols) / 255.0


def _read_labels(path):
    """Read an IDX label file into an array of shape [n_labels]."""
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, "Not an MNIST label file"
        buf = f.read(num)
        return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def load_mnist():
    """Download (if needed) and return
    (X_train, y_train, X_test, y_test)

    X_* : float32 pixels in [0, 1], shape [n, 784]
    y_* : integer labels 0-9,       shape [n]
    """
    X_train = _read_images(_download(FILES["train_images"]))
    y_train = _read_labels(_download(FILES["train_labels"]))
    X_test = _read_images(_download(FILES["test_images"]))
    y_test = _read_labels(_download(FILES["test_labels"]))
    return X_train, y_train, X_test, y_test


def one_hot(labels, num_classes=10):
    """Turn integer labels [3, 0, 1, ...] into one-hot rows, e.g.
    3 -> [0,0,0,1,0,0,0,0,0,0].  Shape: [n, num_classes]."""
    onehot = np.zeros((labels.shape[0], num_classes))
    onehot[np.arange(labels.shape[0]), labels] = 1.0
    return onehot


if __name__ == "__main__":
    # Quick check that loading works and shapes are as expected.
    Xtr, ytr, Xte, yte = load_mnist()
    print("train images:", Xtr.shape, "labels:", ytr.shape)
    print("test  images:", Xte.shape, "labels:", yte.shape)
    print("pixel range:", Xtr.min(), "to", Xtr.max())
    print("first 10 training labels:", ytr[:10])
