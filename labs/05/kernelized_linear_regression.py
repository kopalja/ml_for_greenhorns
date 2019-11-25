#!/usr/bin/env python3
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

def rbf(x, y, gamma, deg):
    # deg not used, included just to unify API with poly kernel
    return np.exp(-gamma * (np.linalg.norm(x - y) ** 2))

def poly(x, y, gamma, deg):
    return (gamma * np.dot(x, y) + 1) ** deg

def predict(coefs, z, kernel_f, examples, bias):
    return np.sum([coefs[i] * kernel_f(z, examples[i], args.kernel_gamma, args.kernel_degree) for i in range(len(examples))]) + bias

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", default=50, type=int, help="Number of examples")
    parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
    parser.add_argument("--kernel_degree", default=3, type=int, help="Degree for poly kernel")
    parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
    parser.add_argument("--iterations", default=100, type=int, help="Number of training iterations")
    parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization weight")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--plot", default=True, action="store_true", help="Plot progress")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Generate an artifical regression dataset
    train_data = np.linspace(-1, 1, args.examples)
    train_targets = np.sin(5 * train_data) + np.random.normal(scale=0.25, size=args.examples) + 1

    test_data = np.linspace(-1.2, 1.2, 2 * args.examples)
    test_targets = np.sin(5 * test_data)+ 1

    coefs = np.zeros(args.examples)
    bias = np.mean(train_targets)

    kernel_f = rbf if args.kernel == 'rbf' else poly 
    kernel = np.zeros([args.examples, args.examples])
    for i in range(args.examples):
        for j in range(args.examples):
            kernel[i, j] = kernel_f(train_data[i], train_data[j], args.kernel_gamma, args.kernel_degree)

    # TODO: Perform `iterations` of SGD-like updates, but in dual formulation
    # using `coefs` as weights of individual training examples.
    #
    # We assume the primary formulation of our model is
    #   y = phi(x)^T w + bias
    # and the loss in the primary problem is MSE with L2 regularization:
    #   L = sum_{i=1}^N [1/2 * (target_i - phi(x_i)^T w - bias)^2] + 1/2 * args.l2 * w^2
    #
    # For bias use explicitly the average of training targets, and do not update
    # it futher during training.
    #
    # Instead of using feature map `phi` directly, we use given kernel computing
    #   K(x, y) = phi(x)^T phi(y)
    # We consider the following kernels:
    # - poly: K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - rbf: K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    #
    # After each update print RMSE both on training and testing data.
    for iteration in range(args.iterations):
        coefs *= (1 - args.learning_rate * args.l2)
        coefs += args.learning_rate * (train_targets - kernel @ coefs - bias)

        train_predictions = [predict(coefs, z, kernel_f, train_data, bias) for z in train_data]
        test_predictions = [predict(coefs, z, kernel_f, train_data, bias) for z in test_data]
        print("Iteration {}, train RMSE {:.2f}, test RMSE {:.2f}".format(
            iteration + 1,
            np.sqrt(sklearn.metrics.mean_squared_error(train_targets, train_predictions)),# RMSE on train data,
            np.sqrt(sklearn.metrics.mean_squared_error(test_targets, test_predictions)),
        ))

    if args.plot:
        test_predictions = [predict(coefs, z, kernel_f, train_data, bias) for z in test_data]

        plt.plot(train_data, train_targets, "bo", label="Train targets")
        plt.plot(test_data, test_targets, "ro", label="Test targets")
        plt.plot(test_data, test_predictions, "g-", label="Predictions")
        plt.legend(loc="upper left")
        plt.show()
