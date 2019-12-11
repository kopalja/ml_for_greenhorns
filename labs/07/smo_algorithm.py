#!/usr/bin/env python3
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection



def model(x, train_data, train_target, a, b):
    return sum([a[i] * train_target[i] * kernel(train_data[i], x) for i in range(len(a))]) + b

def predict(x):
    return sum(a[i] * train_target[i] * kernel(train_data[i], x) for i in range(len(a))) + b


def decision_function(alphas, target, kernel, X_train, x_test, b):
    result = (alphas * target) @ kernel(X_train, x_test) - b
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", default=1, type=float, help="Inverse regularization strenth")
    parser.add_argument("--examples", default=200, type=int, help="Number of examples")
    parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
    parser.add_argument("--kernel_degree", default=5, type=int, help="Degree for poly kernel")
    parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
    parser.add_argument("--num_passes", default=10, type=int, help="Number of passes without changes to stop after")
    parser.add_argument("--plot", default=True, action="store_true", help="Plot progress")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_ratio", default=0.5, type=float, help="Test set size ratio")
    parser.add_argument("--tolerance", default=1e-4, type=float, help="Default tolerance for KKT conditions")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Generate an artifical regression dataset, with +-1 as targets
    data, target = sklearn.datasets.make_classification(
        n_samples=args.examples, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    target = 2 * target - 1

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_ratio` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_ratio, random_state=args.seed)

    # We consider the following kernels:
    # - linear: K(x, y) = x^T y
    # - poly: K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - rbf: K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    def kernel(x, y):
        if args.kernel == "linear":
            return x @ y
        if args.kernel == "poly":
            return (args.kernel_gamma * x @ y + 1) ** args.kernel_degree
        if args.kernel == "rbf":
            return np.exp(-args.kernel_gamma * ((x - y) @ (x - y)))




    # Create initial weights#!/usr/bin/env python3
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection



def predict(x):
    return sum(a[i] * train_target[i] * kernel(train_data[i], x) for i in range(len(a))) + b




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", default=1, type=float, help="Inverse regularization strenth")
    parser.add_argument("--examples", default=200, type=int, help="Number of examples")
    parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
    parser.add_argument("--kernel_degree", default=5, type=int, help="Degree for poly kernel")
    parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
    parser.add_argument("--num_passes", default=10, type=int, help="Number of passes without changes to stop after")
    parser.add_argument("--plot", default=True, action="store_true", help="Plot progress")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_ratio", default=0.5, type=float, help="Test set size ratio")
    parser.add_argument("--tolerance", default=1e-4, type=float, help="Default tolerance for KKT conditions")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Generate an artifical regression dataset, with +-1 as targets
    data, target = sklearn.datasets.make_classification(
        n_samples=args.examples, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    target = 2 * target - 1

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_ratio` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_ratio, random_state=args.seed)

    # We consider the following kernels:
    # - linear: K(x, y) = x^T y
    # - poly: K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - rbf: K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    def kernel(x, y):
        if args.kernel == "linear":
            return x @ y
        if args.kernel == "poly":
            return (args.kernel_gamma * x @ y + 1) ** args.kernel_degree
        if args.kernel == "rbf":
            return np.exp(-args.kernel_gamma * ((x - y) @ (x - y)))




    # Create initial weights
    a, b = np.zeros(len(train_data)), 0
    j_generator = np.random.RandomState(args.seed)
    passes = 0
    E = [predict(train_data[i]) - train_target[i] for i in range(len(train_data))]

    #E = np.array([1 if i > 0 else -1 for i in r])
    #E = [0 for i in range(len(train_data))]
    while passes < args.num_passes:
        a_changed = 0
        for i in range(len(a)):
            # TODO: Check that a[i] fulfuls the KKT condition, using args.tolerance for comparing predictions with zero.
            # If the conditions, do not hold, then
            # - choose random j as
            #     j = j_generator.randint(len(a) - 1)
            #     j = j + (j >= i)
            #
            # - compute the updated unclipped a_j^new.
            #   Note that if the second derivative of the loss with respect
            #   to a[j] is >= -args.tolerance, do not update a[j] and
            #   continue with next i.
            #
            # - clip the a_j^new to [L, H].
            #   If the clip window is too narrow (H - L < args.tolerance), keep
            #   the original a[j] and continue with next i; also keep the
            #   original a[j] if the clipped updated a_j^new does not differ
            #   from the original a[j] by more than args.tolerance.
            #
            # - update a[j] to a_j^new, and compute the updated a[i] and b
            #
            # - increase a_changed
            if (a[i] < args.C and train_target[i] * E[i] < - args.tolerance) or (a[i] > 0 and train_target[i] * E[i] > args.tolerance):
                j = j_generator.randint(len(a) - 1)
                j = j + (j >= i)
                k11 = kernel(train_data[i], train_data[i])
                k12 = kernel(train_data[i], train_data[j])
                k22 = kernel(train_data[j], train_data[j])
                eta = 2 * k12 - k11 - k22
                if (eta > -args.tolerance):
                    continue
                L = max(0, a[j] - a[i]) if train_target[i] != train_target[j] else max(0, a[i] + a[j] - args.C)
                H = min(args.C, args.C + a[j] - a[i]) if train_target[i] != train_target[j] else  min(args.C, a[i] + a[j])
                if (H - L < args.tolerance):
                    continue
                # compute a_j
                a_j_new = a[j] - train_target[j] * (E[i] - E[j]) / eta
                a_j_new = min(H, a_j_new)
                a_j_new = max(L, a_j_new)
                if (np.abs(a[j] - a_j_new) < args.tolerance):
                    continue
                a_i_new = a[i] + (train_target[i] * train_target[j]) * (a[j] - a_j_new)
                b1 = E[i] + train_target[i] * (a_i_new - a[i]) * k11 + train_target[j] * (a_j_new - a[j]) * k12 + b
                b2 = E[j] + train_target[j] * (a_i_new - a[i]) * k12 + train_target[j] * (a_j_new - a[j]) * k22 + b
                if 0 < a_i_new and a_i_new < args.C:
                    b = b1
                elif 0 < a_j_new and a_j_new < args.C:
                    b = b2
                else:
                    b = (b1 + b2) * 0.5
                a[i] = a_i_new
                a[j] = a_j_new
                a_changed += 1

        passes = 0 if a_changed else passes + 1





        # TODO: After each iteration, measure the accuracy for both the
        #train test and the test set and print it in percentages.
        predictions = [1 if predict(example) > 0 else -1 for example in train_data]
        train_accuracy = sum([1 if r == t else 0 for r, t in zip(predictions, train_target)]) / len(predictions)

        predictions = [1 if predict(example) > 0 else -1 for example in test_data]
        test_accuracy = sum([1 if r == t else 0 for r, t in zip(predictions, test_target)]) / len(predictions)

        print("Train acc {:.1f}%, test acc {:.1f}%".format(
            100 * train_accuracy,
            100 * test_accuracy,
        ))

    if args.plot:
        def predict(x):
            return sum(a[i] * train_target[i] * kernel(train_data[i], x) for i in range(len(a))) + b
        xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
        ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
        predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
        plt.contourf(xs, ys, predictions, levels=0, cmap=plt.cm.RdBu)
        plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
        plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap=plt.cm.RdBu, zorder=2)
        plt.scatter(train_data[a > args.tolerance, 0], train_data[a > args.tolerance, 1], marker="o", s=90, label="Support Vectors", c="#00dd00")
        plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap=plt.cm.RdBu, zorder=2)
        plt.legend(loc="upper center", ncol=3)
        plt.show()
