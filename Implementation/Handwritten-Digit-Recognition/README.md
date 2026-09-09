# Handwritten Digit Recognition — Neural Network from Scratch

Recognise handwritten digits (0–9) from the **MNIST** dataset using a neural
network whose **backpropagation is written by hand** — no TensorFlow, PyTorch,
Keras, or scikit-learn. The only third-party package is **NumPy**, used purely
for fast array arithmetic (matrix multiplication, etc.). The code is written for
**complete beginners** and organised with **classes and objects**.

---

## What is this?

We train a small **multilayer perceptron** (a basic neural network):

```
784 inputs  ->  128 hidden neurons (ReLU)  ->  10 outputs (softmax)
(28x28 pixels)                                  (one score per digit 0-9)
```

Given a 28×28 grey-scale image of a digit (flattened into 784 numbers), the
network outputs 10 probabilities; the highest one is its guess.

It learns by **gradient descent**: make a prediction (forward pass), measure the
error (cross-entropy loss), compute how each weight should change
(**backpropagation**), and nudge every weight a little. Repeat many times.

---

## Files

| File | What it does |
|------|--------------|
| `activations.py` | Activation functions as classes: `ReLU`, `Sigmoid`, `Softmax` (each bundles the function and its derivative). |
| `neural_network.py` | The core. `DenseLayer` (one layer) and `NeuralNetwork` (the whole model): forward pass, **hand-written backprop**, training, prediction, save/load. |
| `mnist_loader.py` | Downloads MNIST once and reads its binary format with the standard library + NumPy. |
| `train.py` | Loads data, builds the network, trains it, prints accuracy, saves `model.npz`. |
| `predict.py` | Loads `model.npz` and shows predictions for a few test images (drawn in the terminal). |
| `sanity_check.py` | A fast, download-free test proving the backprop actually learns. |

---

## How to run

1. **Install NumPy** (the only requirement):

   ```bash
   pip install -r requirements.txt
   ```

2. **Check that the maths works** (fast, no download):

   ```bash
   python sanity_check.py
   ```

   You should see the loss fall to almost zero and `SANITY CHECK PASSED`.

3. **Train on MNIST** (downloads ~11 MB the first time):

   ```bash
   python train.py
   ```

   Training all 15 epochs takes a few minutes on a normal laptop CPU and
   reaches roughly **97–98% test accuracy**. It saves the weights to
   `model.npz`.

4. **See it predict**:

   ```bash
   python predict.py
   ```

---

## The maths, briefly

For each layer, with input `a_prev`, weights `W`, bias `b`, and activation `φ`:

**Forward**
```
z = a_prev · W + b        (weighted sum + bias)
a = φ(z)                  (activation)
```

**Output layer** uses **softmax** to make probabilities, and the loss is
**cross-entropy**:
```
E = −Σ  y · log(ŷ)        (y = true one-hot label, ŷ = predicted probability)
```

**Backpropagation** (the part we implement by hand) sends the error backward:
```
output layer :  dz = ŷ − y                        (softmax + cross-entropy)
any layer    :  dW      = a_prevᵀ · dz / m
                db      = sum(dz) / m
                da_prev = dz · Wᵀ                  (passed to the layer below)
hidden layer :  dz_prev = da_prev ⊙ φ'(z_prev)    (⊙ = element-wise product)
```

**Update** (gradient descent):
```
W ← W − η · dW      b ← b − η · db        (η = learning rate)
```

That is the entire algorithm — see `neural_network.py`, where each of these
lines has a matching, commented line of code.

---

## Ideas to explore

- Change the hidden size (`[784, 256, 10]`) or add a layer (`[784, 128, 64, 10]`).
- Switch the hidden activation to `sigmoid` in `train.py`.
- Try different learning rates (0.01, 0.5) and watch the loss.
