"""
predict.py
----------
Load the trained weights (model.npz) and predict digits for a few test images.

Run it with:
    python predict.py

It prints, for the first few test images, the digit the network predicts and
the digit that is actually correct, and an overall accuracy.  It also draws
each little image in the terminal using text, so you can "see" the digit.
"""

import numpy as np

from mnist_loader import load_mnist
from neural_network import NeuralNetwork


def ascii_image(pixels):
    """Turn one 784-length image into a small block of text so we can look
    at the digit right in the terminal."""
    chars = " .:-=+*#%@"          # from light to dark
    img = pixels.reshape(28, 28)
    lines = []
    for row in img:
        line = "".join(chars[min(int(p * len(chars)), len(chars) - 1)] for p in row)
        lines.append(line)
    return "\n".join(lines)


def main(num_examples=5):
    # Load the data and the network with the SAME architecture used in training.
    X_train, y_train, X_test, y_test = load_mnist()
    net = NeuralNetwork([784, 128, 10])
    net.load("model.npz")

    # Overall accuracy on the whole test set.
    acc = net.evaluate(X_test, y_test)
    print(f"Test accuracy: {acc * 100:.2f}%\n")

    # Show a few individual predictions.
    preds = net.predict(X_test[:num_examples])
    for i in range(num_examples):
        print(ascii_image(X_test[i]))
        mark = "correct" if preds[i] == y_test[i] else "WRONG"
        print(f"  predicted = {preds[i]}   actual = {y_test[i]}   ({mark})\n")


if __name__ == "__main__":
    main()
