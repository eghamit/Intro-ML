"""
train.py
--------
Train the from-scratch neural network on MNIST and save the learned weights.

Run it with:
    python train.py

What it does:
    1. Loads the MNIST data (downloading it the first time).
    2. Builds a 784 -> 128 -> 10 network.
    3. Trains it with mini-batch gradient descent (backprop by hand).
    4. Prints the test accuracy and saves the weights to model.npz.
"""

from mnist_loader import load_mnist, one_hot
from neural_network import NeuralNetwork


def main():
    # 1. Load the data ------------------------------------------------------
    print("Loading MNIST ...")
    X_train, y_train, X_test, y_test = load_mnist()
    y_train_oh = one_hot(y_train)   # one-hot targets for training

    # 2. Build the network --------------------------------------------------
    #    784 input pixels -> 128 hidden neurons (ReLU) -> 10 outputs (softmax)
    net = NeuralNetwork([784, 128, 10], hidden_activation="relu")

    # 3. Train --------------------------------------------------------------
    print("Training ...")
    net.train(
        X_train, y_train_oh,
        epochs=15,
        batch_size=64,
        learning_rate=0.1,
        X_val=X_test, y_val=y_test,   # show test accuracy each epoch
        verbose=True,
    )

    # 4. Final evaluation + save -------------------------------------------
    acc = net.evaluate(X_test, y_test)
    print(f"\nFinal test accuracy: {acc * 100:.2f}%")

    net.save("model.npz")
    print("Saved trained weights to model.npz")


if __name__ == "__main__":
    main()
