#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b

#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection



def preprocess_data(combination, data, target):
    r_data, r_target = [], []
    for d, t in zip(data, target):
        if t == combination[0]:
            r_target.append(-1)
            r_data.append(d)
        elif t == combination[1]:
            r_target.append(1)
            r_data.append(d)
    return np.array(r_data), np.array(r_target)
        


def smo(train_data, train_target, test_data, args):
    # TODO: Use exactly the SMO algorithm from `smo_algorithm` assignment.
    #
    # The `j_generator` should be created every time with the same seed.
    # Create initial weights

    def kernel(x, y):
        if args.kernel == "linear":
            return x @ y
        if args.kernel == "poly":
            return (args.kernel_gamma * x @ y + 1) ** args.kernel_degree
        if args.kernel == "rbf":
            return np.exp(-args.kernel_gamma * ((x - y) @ (x - y)))


    def predict(x):
        return sum(a[i] * train_target[i] * kernel(train_data[i], x) for i in range(len(a))) + b


    a, b = np.zeros(len(train_data)), 0
    j_generator = np.random.RandomState(args.seed)
    passes = 0
    E = [predict(train_data[i]) - train_target[i] for i in range(len(train_data))]

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
            E[i] = predict(train_data[i]) - train_target[i]
            if (a[i] < args.C and train_target[i] * E[i] < - args.tolerance) or (a[i] > 0 and train_target[i] * E[i] > args.tolerance):
                j = j_generator.randint(len(a) - 1)
                j = j + (j >= i)
                E[j] = predict(train_data[j]) - train_target[j]
                k11 = kernel(train_data[i], train_data[i])
                k12 = kernel(train_data[i], train_data[j])
                k22 = kernel(train_data[j], train_data[j])
                k21 = kernel(train_data[j], train_data[i])
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
                b_j = b - E[j] - train_target[i] * (a_i_new - a[i]) * k12 - train_target[j] * (a_j_new - a[j]) * k22
                b_i = b - E[i] - train_target[i] * (a_i_new - a[i]) * k11 - train_target[j] * (a_j_new - a[j]) * k21
                if 0 < a_i_new and a_i_new < args.C:
                    b = b_i
                elif 0 < a_j_new and a_j_new < args.C:
                    b = b_j
                else:
                    b = (b_i + b_j) * 0.5
                a[i] = a_i_new
                a[j] = a_j_new
                a_changed += 1

        passes = 0 if a_changed else passes + 1
    
    return [1 if predict(example) > 0 else 0 for example in test_data]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", default=1, type=float, help="Inverse regularization strenth")
    parser.add_argument("--classes", default=4, type=int, help="Number of classes")
    parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
    parser.add_argument("--kernel_degree", default=5, type=int, help="Degree for poly kernel")
    parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
    parser.add_argument("--num_passes", default=10, type=int, help="Number of passes without changes to stop after")
    parser.add_argument("--plot", default=False, action="store_true", help="Plot progress")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=701, type=int, help="Test set size")
    parser.add_argument("--tolerance", default=1e-4, type=float, help="Default tolerance for KKT conditions")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load the digits dataset with specified number of classes, and normalize it.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    data /= np.max(data)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_ratio` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)



    # TODO: Using One-vs-One scheme, train (K \binom 2) classifiers, one for every
    # pair of classes.

    # Then, classify the test set by majority voting, using the lowest class
    # index in case of ties. Finally compute `test accuracy`.
    votes = np.zeros(shape = (len(test_data), args.classes))
    for combination in [(i, j) for i in range(args.classes) for j in range(i + 1, args.classes)]:
        data, target = preprocess_data(combination, train_data, train_target)
        pair_results = smo(data, target, test_data, args)
        for k in range(len(pair_results)):
            votes[k][combination[0]] += 1 - pair_results[k]
            votes[k][combination[1]] += pair_results[k]


    predictions = [np.argmax(v) for v in votes]
    test_accuracy = sum([1 if r == t else 0 for r, t in zip(predictions, test_target)]) / len(predictions)


    print("{:.2f}".format(100 * test_accuracy))
