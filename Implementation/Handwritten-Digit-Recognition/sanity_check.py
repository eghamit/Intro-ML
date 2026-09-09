"""
sanity_check.py
---------------
A tiny, fast test that needs NO download.  It trains the network on a small
made-up dataset and checks that the loss goes DOWN and the network learns to
classify it almost perfectly.  If this passes, the backprop math is working.

Run it with:
    python sanity_check.py
"""

import numpy as np

from neural_network import NeuralNetwork
from mnist_loader import one_hot


def main():
    rng = np.random.default_rng(42)

    # Make 300 fake examples in 20 dimensions belonging to 3 classes.
    # Each class is a blob centred on its own random point, so the classes
    # are separable and a working network should nail them.
    n_per_class, dim, n_classes = 100, 20, 3
    centres = rng.standard_normal((n_classes, dim)) * 3.0
    X, y = [], []
    for c in range(n_classes):
        X.append(centres[c] + rng.standard_normal((n_per_class, dim)))
        y.append(np.full(n_per_class, c))
    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y)
    y_oh = one_hot(y, num_classes=n_classes)

    net = NeuralNetwork([dim, 16, n_classes])

    # Loss before training.
    start_loss = net.cross_entropy(net.forward(X), y_oh)
    net.train(X, y_oh, epochs=40, batch_size=32, learning_rate=0.2, verbose=False)
    end_loss = net.cross_entropy(net.forward(X), y_oh)
    acc = net.evaluate(X, y)

    print(f"loss before training : {start_loss:.4f}")
    print(f"loss after  training : {end_loss:.4f}")
    print(f"training accuracy    : {acc * 100:.2f}%")

    assert end_loss < start_loss, "Loss did not decrease -- backprop is broken!"
    assert acc > 0.95, "Accuracy too low -- something is wrong."
    print("\nSANITY CHECK PASSED: the network learns, so backprop works.")


if __name__ == "__main__":
    main()
