"""
activations.py
--------------
Activation functions, written as small CLASSES so that each one bundles
together *both* the function itself (``forward``) and its derivative
(``derivative``).  Grouping the two together is convenient for
backpropagation, which needs the derivative.

Everything here is plain NumPy -- there is no machine-learning library doing
the work for us.  NumPy is only used for fast array arithmetic.

A beginner's mental model:
    * ``forward(z)``     : squashes the raw neuron value ``z`` into ``a``.
    * ``derivative(z)``  : the slope of that squashing, needed by backprop.
"""

import numpy as np


class ReLU:
    """Rectified Linear Unit:  f(z) = max(0, z).

    The most common hidden-layer activation.  It simply keeps positive
    values and turns negative values into zero.
    """

    def forward(self, z):
        return np.maximum(0.0, z)

    def derivative(self, z):
        # Slope is 1 where z was positive, and 0 where z was negative.
        return (z > 0).astype(z.dtype)


class Sigmoid:
    """Logistic sigmoid:  f(z) = 1 / (1 + e^{-z}).

    Squashes any number into the range (0, 1).  Included so you can
    experiment with a different hidden activation.
    """

    def forward(self, z):
        # np.clip keeps the exponent in a safe range to avoid overflow.
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def derivative(self, z):
        s = self.forward(z)
        return s * (1.0 - s)


class Softmax:
    """Softmax turns a vector of scores into a probability distribution
    (all entries positive and summing to 1).  Used on the OUTPUT layer for
    multi-class classification (here: the 10 digit classes 0-9).

    Note: we do not need a standalone ``derivative`` here.  When softmax is
    paired with the cross-entropy loss, the gradient collapses to the very
    simple form ``(predicted_probabilities - true_labels)`` -- this is
    handled directly inside the network's backward pass.
    """

    def forward(self, z):
        # Subtract the row max before exponentiating: mathematically
        # identical, but numerically stable (prevents huge exponentials).
        z_shift = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z_shift)
        return exp / np.sum(exp, axis=1, keepdims=True)
