"""
mnist_loader.py
---------------
Loads the MNIST handwritten-digit dataset FROM LOCAL FILES ONLY.

This loader never downloads anything.  You place the four MNIST files on disk
yourself (once), and this script just reads them.

Expected folder layout (put the files inside a sub-folder of ``data/``):

    Handwritten-Digit-Recognition/
        data/
            MNIST/                         <- any name works (MNIST, Fashion, ...)
                train-images-idx3-ubyte.gz   60000 training images
                train-labels-idx1-ubyte.gz   60000 training labels
                t10k-images-idx3-ubyte.gz    10000 test images
                t10k-labels-idx1-ubyte.gz    10000 test labels
        mnist_loader.py

Both the compressed ``.gz`` files and the uncompressed ``idx-ubyte`` files are
accepted.  MNIST is stored in a simple binary "IDX" format, which we parse by
hand with only the Python standard library (gzip, struct) plus NumPy.

Usage
-----
    from mnist_loader import load_mnist
    X_train, y_train, X_test, y_test = load_mnist()
"""

import gzip
import os
import struct

import numpy as np

# The logical name of each file -> the base filename to look for on disk.
FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte",
}

# The "data" folder sits next to this script.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(_HERE, "data")


def _candidate_dirs(data_dir):
    """Directories to search for the MNIST files, in order:
        1. the explicit ``data_dir`` the caller gave (if any),
        2. ``data/`` itself,
        3. every immediate sub-folder of ``data/`` (e.g. data/MNIST, data/Fashion).
    This way it does not matter what you named the sub-folder.
    """
    dirs = []
    if data_dir is not None:
        dirs.append(data_dir)
    dirs.append(DATA_ROOT)
    if os.path.isdir(DATA_ROOT):
        for name in sorted(os.listdir(DATA_ROOT)):
            sub = os.path.join(DATA_ROOT, name)
            if os.path.isdir(sub):
                dirs.append(sub)
    return dirs


def _find(base_name, data_dir):
    """Find ``base_name`` (with or without a .gz suffix) in the candidate
    directories.  Return its full path, or raise a helpful error."""
    for d in _candidate_dirs(data_dir):
        for fname in (base_name + ".gz", base_name):
            path = os.path.join(d, fname)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(
        f"\nCould not find '{base_name}(.gz)'.\n"
        f"This project does NOT download data. Please place the four MNIST files:\n"
        f"    {FILES['train_images']}.gz\n"
        f"    {FILES['train_labels']}.gz\n"
        f"    {FILES['test_images']}.gz\n"
        f"    {FILES['test_labels']}.gz\n"
        f"inside a sub-folder of:\n    {DATA_ROOT}\n"
        f"e.g.  {os.path.join(DATA_ROOT, 'MNIST')}\n"
        f"(You can obtain them from any MNIST mirror, e.g. "
        f"https://storage.googleapis.com/cvdf-datasets/mnist/)"
    )


def _open(path):
    """Open a file, transparently handling gzip compression."""
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def _read_images(path):
    """Read an IDX image file into an array of shape [n_images, 784]."""
    with _open(path) as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"{path} is not an MNIST image file"
        buf = f.read(rows * cols * num)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # Flatten each 28x28 image to a 784-vector, scale pixels to [0, 1].
        return data.reshape(num, rows * cols) / 255.0


def _read_labels(path):
    """Read an IDX label file into an array of shape [n_labels]."""
    with _open(path) as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"{path} is not an MNIST label file"
        buf = f.read(num)
        return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def load_mnist(data_dir=None):
    """Read the local MNIST files and return
    (X_train, y_train, X_test, y_test)

    X_* : float32 pixels in [0, 1], shape [n, 784]
    y_* : integer labels 0-9,       shape [n]

    Pass ``data_dir`` to point directly at the folder that holds the files;
    otherwise it is auto-detected under ``data/``.
    """
    X_train = _read_images(_find(FILES["train_images"], data_dir))
    y_train = _read_labels(_find(FILES["train_labels"], data_dir))
    X_test = _read_images(_find(FILES["test_images"], data_dir))
    y_test = _read_labels(_find(FILES["test_labels"], data_dir))
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
