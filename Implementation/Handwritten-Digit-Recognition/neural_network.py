"""
neural_network.py
-----------------
A neural network for recognising handwritten digits, built FROM SCRATCH.

The forward pass and -- most importantly -- the BACKPROPAGATION are written
out by hand using only NumPy.  No deep-learning library (TensorFlow, PyTorch,
Keras, scikit-learn's MLP, ...) is used.

The code is organised with classes and objects (OOP):

    DenseLayer      -- one fully-connected layer (its weights, bias, and the
                       forward/backward math for that layer).
    NeuralNetwork   -- stacks several DenseLayers, and runs training, the
                       backward pass, prediction and evaluation.

------------------------------------------------------------------------------
THE MATH (for a single layer l), using varphi for the activation function:

    forward :   z = a_prev @ W + b          (weighted sum + bias)
                a = varphi(z)               (activation)

    backward:   given dL/dz for this layer,
                dL/dW      = a_prev^T @ (dL/dz) / m
                dL/db      = sum(dL/dz) / m
                dL/da_prev = (dL/dz) @ W^T        <- passed to the layer below

For the OUTPUT layer we use softmax + cross-entropy, whose combined gradient
is simply:   dL/dz = (predicted_probabilities - true_one_hot_labels).
------------------------------------------------------------------------------
"""

import numpy as np
from activations import ReLU, Sigmoid, Softmax


class DenseLayer:
    """A single fully-connected ("dense") layer.

    It owns a weight matrix ``W`` of shape (n_inputs, n_neurons) and a bias
    row-vector ``b`` of shape (1, n_neurons), plus an activation object.
    """

    def __init__(self, n_inputs, n_neurons, activation):
        # ---- Weight initialisation ----
        # "He initialisation": random small numbers scaled by sqrt(2/n_inputs).
        # Good defaults for ReLU; breaks symmetry so neurons learn different
        # things instead of all staying identical.
        rng = np.random.default_rng()  # NumPy's random generator
        self.W = rng.standard_normal((n_inputs, n_neurons)) * np.sqrt(2.0 / n_inputs)
        self.b = np.zeros((1, n_neurons))
        self.activation = activation

        # These caches are filled during the forward pass and reused by the
        # backward pass (backprop reuses forward-pass values -- that is the
        # whole efficiency trick).
        self.a_prev = None   # input to this layer  (activations of layer below)
        self.z = None        # pre-activation (weighted sum + bias)
        self.a = None        # output activation

        # Gradients, computed during the backward pass.
        self.dW = None
        self.db = None

    def forward(self, a_prev):
        """Compute this layer's output for a batch of inputs ``a_prev``
        (shape: [batch_size, n_inputs])."""
        self.a_prev = a_prev
        self.z = a_prev @ self.W + self.b     # weighted sum + bias
        self.a = self.activation.forward(self.z)
        return self.a

    def backward(self, grad_z, m):
        """Given ``grad_z`` = dL/dz for THIS layer (shape [batch, n_neurons]),
        compute and store this layer's weight/bias gradients, and return
        dL/da_prev to hand down to the previous layer.

        ``m`` is the batch size, used to average the gradient over the batch.
        """
        self.dW = self.a_prev.T @ grad_z / m
        self.db = np.sum(grad_z, axis=0, keepdims=True) / m
        grad_a_prev = grad_z @ self.W.T
        return grad_a_prev

    def update(self, learning_rate):
        """Take one gradient-descent step on this layer's parameters."""
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db


class NeuralNetwork:
    """A multilayer perceptron for classification.

    Example
    -------
    >>> net = NeuralNetwork([784, 128, 10])   # 784 inputs, 128 hidden, 10 out
    >>> net.train(X_train, y_train, epochs=10, batch_size=64, learning_rate=0.1)
    >>> accuracy = net.evaluate(X_test, y_test)
    """

    def __init__(self, layer_sizes, hidden_activation="relu"):
        """``layer_sizes`` e.g. [784, 128, 10]:
        input size, then one number per layer (hidden..., output).
        Hidden layers use ReLU (or sigmoid); the output layer uses softmax.
        """
        self.layer_sizes = layer_sizes
        hidden = ReLU() if hidden_activation == "relu" else Sigmoid()

        # Build the list of layers.
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            is_output = (i == len(layer_sizes) - 2)
            act = Softmax() if is_output else hidden
            self.layers.append(DenseLayer(n_in, n_out, act))

    # ------------------------------------------------------------------ #
    #  Forward pass                                                       #
    # ------------------------------------------------------------------ #
    def forward(self, X):
        """Run inputs ``X`` (shape [batch, 784]) through every layer and
        return the output probabilities (shape [batch, 10])."""
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a  # these are softmax probabilities

    # ------------------------------------------------------------------ #
    #  Loss                                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def cross_entropy(probs, y_onehot):
        """Average cross-entropy loss over the batch.
        ``probs``    : predicted probabilities, shape [batch, 10]
        ``y_onehot`` : true labels as one-hot rows,  shape [batch, 10]
        """
        m = y_onehot.shape[0]
        # add a tiny epsilon inside the log to avoid log(0)
        eps = 1e-12
        return -np.sum(y_onehot * np.log(probs + eps)) / m

    # ------------------------------------------------------------------ #
    #  Backward pass  (backpropagation, implemented by hand)              #
    # ------------------------------------------------------------------ #
    def backward(self, y_onehot):
        """Compute the gradient of the loss w.r.t. every weight and bias.
        Assumes ``forward`` has just been called (caches are populated)."""
        m = y_onehot.shape[0]

        # Output layer: softmax + cross-entropy gives this clean gradient.
        probs = self.layers[-1].a
        grad_z = probs - y_onehot            # dL/dz at the output layer

        # Walk backwards through the layers.
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            grad_a_prev = layer.backward(grad_z, m)   # fills layer.dW, layer.db
            if i > 0:
                # Convert dL/da_prev into dL/dz for the previous (hidden) layer
                # by multiplying with that layer's activation derivative.
                prev = self.layers[i - 1]
                grad_z = grad_a_prev * prev.activation.derivative(prev.z)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    # ------------------------------------------------------------------ #
    #  Training loop  (mini-batch gradient descent)                       #
    # ------------------------------------------------------------------ #
    def train(self, X, y_onehot, epochs=10, batch_size=64,
              learning_rate=0.1, X_val=None, y_val=None, verbose=True):
        """Train with mini-batch stochastic gradient descent.

        One *epoch* = one full pass over the training data, done in small
        random *batches* so each weight update is fast.
        """
        n = X.shape[0]
        rng = np.random.default_rng(0)
        for epoch in range(1, epochs + 1):
            # Shuffle the data at the start of each epoch.
            order = rng.permutation(n)
            X_shuf, y_shuf = X[order], y_onehot[order]

            # Loop over mini-batches.
            for start in range(0, n, batch_size):
                xb = X_shuf[start:start + batch_size]
                yb = y_shuf[start:start + batch_size]

                self.forward(xb)          # 1. forward pass
                self.backward(yb)         # 2. backprop -> gradients
                self.update(learning_rate)  # 3. gradient-descent step

            if verbose:
                probs = self.forward(X)
                loss = self.cross_entropy(probs, y_onehot)
                msg = f"Epoch {epoch:2d}/{epochs}  loss={loss:.4f}"
                if X_val is not None:
                    msg += f"  val_acc={self.evaluate(X_val, y_val):.4f}"
                print(msg)

    # ------------------------------------------------------------------ #
    #  Prediction & evaluation                                            #
    # ------------------------------------------------------------------ #
    def predict(self, X):
        """Return the predicted digit (0-9) for each input row."""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y_labels):
        """Return classification accuracy. ``y_labels`` are integer labels."""
        preds = self.predict(X)
        return float(np.mean(preds == y_labels))

    # ------------------------------------------------------------------ #
    #  Saving / loading the trained weights                               #
    # ------------------------------------------------------------------ #
    def save(self, path):
        params = {}
        for i, layer in enumerate(self.layers):
            params[f"W{i}"] = layer.W
            params[f"b{i}"] = layer.b
        np.savez(path, sizes=np.array(self.layer_sizes), **params)

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        for i, layer in enumerate(self.layers):
            layer.W = data[f"W{i}"]
            layer.b = data[f"b{i}"]
