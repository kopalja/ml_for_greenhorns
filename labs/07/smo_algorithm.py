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




    # Create initial weights
    a, b = np.zeros(len(train_data)), 0
    j_generator = np.random.RandomState(args.seed)
    passes = 0
    E = np.array([predict(train_data[i]) - train_target[i] for i in range(len(train_data))])
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

                alph1 = a[i]
                alph2 = a[j]
                y1 = train_target[i]
                y2 = train_target[j]
                E1 = E[i]
                E2 = E[j]
                s = y1 * y2

                L = max(0, alph2 - alph1) if y1 != y2 else max(0, alph1 + alph2 - args.C)
                H = min(args.C, args.C + alph2 - alph1) if y1 != y2 else  min(args.C, alph1 + alph2)

                if (H - L < args.tolerance):
                    continue

                # Compute kernel & 2nd derivative eta
                k11 = kernel(train_data[i], train_data[i])
                k12 = kernel(train_data[i], train_data[j])
                k22 = kernel(train_data[j], train_data[j])
                eta = 2 * k12 - k11 - k22

                if (eta > -args.tolerance):
                    continue

                a2 = alph2 - y2 * (E1 - E2) / eta
                # Clip a2 based on bounds L & H
                a2 = min(H, a2)
                a2 = max(L, a2)
                
                # If examples can't be optimized within epsilon (eps), skip this pair
                if (np.abs(alph2 - a2) < args.tolerance):
                    #print('small change in apha')
                    continue
                
                # compute alpha1
                a1 = alph1 + s * (alph2 - a2)
                

                b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + b
                b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + b

                a[i] = a1
                a[j] = a2
                if 0 < a1 and a1 < args.C:
                    b = b1
                elif 0 < a2 and a2 < args.C:
                    b = b2
                else:
                    b = (b1 + b2) * 0.5
                a_changed += 1

                

        passes = 0 if a_changed else passes + 1





        results = [predict(example) for example in test_data]
        results = [1 if r >= 0 else -1  for r in results]
        train_accuracy = sum([1 if r == t else 0 for r, t in zip(results, test_target)]) / len(results)
        test_accuracy = train_accuracy


        results = [predict(example) for example in train_data]
        results = [1 if r >= 0 else -1  for r in results]
        train_accuracy = sum([1 if r == t else 0 for r, t in zip(results, train_target)]) / len(results)

        # TODO: After each iteration, measure the accuracy for both the
        #train test and the test set and print it in percentages.
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












                # # from pdf
                # E[i] = model(train_data[i], train_data, train_target, a, b) - train_target[i]
                # E[j] = model(train_data[j], train_data, train_target, a, b) - train_target[j]

                # n = kernel(train_data[i], train_data[i]) + kernel(train_data[j], train_data[j]) - 2 * kernel(train_data[i], train_data[j])
                # a_j_new = a[j] + train_target[j] * (E[i] - E[j]) / n

                # L = max(0, a[j] - a[i]) if train_target[i] != train_target[j] else max(0, a[j] + a[i] - args.C)
                # H = min(args.C, args.C + a[j] - a[i]) if train_target[i] != train_target[j] else min(args.C, a[j] + a[i])
                
                # a_j_new = min(a_j_new, H)
                # a_j_new = max(a_j_new, L)

                # s = train_target[i] * train_target[j]
                # a_i_new = a[i] + s * (a[j] - a_j_new)

                # #print('before')
                # # print(n)
                # # if (n >= - args.tolerance) or (H - L < args.tolerance) or (abs(a_j_new - a[j]) < args.tolerance):
                # #     continue
                # # print('update')
                # # b 1 = E 1 + y 1 ( α 1 new − α 1 ) K ( x 1 , x 1 ) + y 2 ( α 2 new,clipped − α 2 ) K ( x 1 , x 2 ) + b .
                # x = (E[i])
                # print(x)
                # b1 = int(E[i] + train_target[i] * (a_i_new - a[i]) * kernel(train_data[i], train_data[i]) + \
                #               train_target[j] * (a_j_new - a[j]) * kernel(train_data[i], train_data[j]) + b)

                # b2 = E[j] + train_target[i] * (a_i_new - a[i]) * kernel(train_data[i], train_data[j]) + \
                #               train_target[j] * (a_j_new - a[j]) * kernel(train_data[j], train_data[j]) + b

                # b = (b2 + b1) / 2
                # a[j] = a_j_new
                # a[i] = a_i_new
                # a_changed += 1





                # if (y1 != y2):
                #     L = max(0, alph2 - alph1)
                #     H = min(args.C, args.C + alph2 - alph1)
                # elif (y1 == y2):
                #     L = max(0, alph1 + alph2 - args.C)
                #     H = min(args.C, alph1 + alph2)

                # if L < a2 < H:
                #     a2 = a2
                # elif (a2 <= L):
                #     a2 = L
                # elif (a2 >= H):
                #     a2 = H



                
                # Update error cache
                # Error cache for optimized alphas is set to 0 if they're unbound
                # for index, alph in zip([i, j], [a1, a2]):
                #     if 0.0 < alph < args.C:
                #         E[index] = 0.0
                
                # # Set non-optimized errors based on equation 12.11 in Platt's book
                # non_opt = [n for n in range(len(train_data)) if (n != i and n != j)]
                # E[non_opt] = E[non_opt] + \
                #                         y1*(a1 - alph1)*kernel(E[i], E[non_opt]) + \
                #                         y2*(a2 - alph2)*kernel(E[j], E[non_opt]) + b - b_new
                
                # Update model threshold