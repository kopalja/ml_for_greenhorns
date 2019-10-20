#!/usr/bin/env python3

#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
    parser.add_argument("--examples", default=1000, type=int, help="Number of examples")
    parser.add_argument("--iterations", default=50, type=int, help="Number of iterations over the data")
    parser.add_argument("--folds", default=10, type=int, help="Number of folds")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--plot", default=False, action="store_true", help="Plot progress")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Generate an artifical regression dataset
    data, target = sklearn.datasets.make_regression(n_samples=args.examples, random_state=args.seed)

    # TODO: Append a constant feature with value 1 to the end of every input data
    data = np.append(data, np.ones((data.shape[0], 1)), axis=1)

    rmses = []
    # TODO: Using `sklearn.model_selection.KFold(args.folds)`, generate the required
    # number of folds. The folds are returned as a generator of
    # (train_data_indices, test_data_indices) pairs.
    folds = sklearn.model_selection.KFold(args.folds)
    for train_indices, test_indices in folds.split(data):
        # TODO: Generate train_data, train_target, test_data, test_target using the fold indices
        train_data, train_target = data[train_indices], target[train_indices]
        test_data, test_target = data[test_indices], target[test_indices]

        # Generate initial linear regression weights
        weights = np.random.uniform(size=train_data.shape[1])

        rmses.append([])
        for iteration in range(args.iterations):
            permutation = np.random.permutation(train_data.shape[0])

            # TODO: Process the data in the order of `permutation`.
            # For every `args.batch_size`, average their gradient, and update the weights.
            # A gradient for example (x_i, t_i) is `(x_i^T weights - t_i) * x_i`,
            # and the SGD update is `weights = weights - learning_rate * gradient`.
            gradient = np.zeros_like(weights)
            for j, i in enumerate(permutation):
                gradient += (np.dot(train_data[i], weights) - train_target[i]) * train_data[i]
                if (j + 1) % args.batch_size == 0:
                    weights -= args.learning_rate * gradient / args.batch_size 
                    gradient = np.zeros_like(weights)

            # We evaluate RMSE on train and test
            rmses[-1].append({
                "train": np.sqrt(sklearn.metrics.mean_squared_error(train_target, train_data @ weights)),
                "test": np.sqrt(sklearn.metrics.mean_squared_error(test_target, test_data @ weights)),
            })

        # TODO: Compute into `explicit_rmse` test data RMSE when
        # fitting `sklearn.linear_model.LinearRegression` on train_data.
        model = sklearn.linear_model.LinearRegression()
        model.fit(train_data, train_target)
        explicit_rmse = np.sqrt(sklearn.metrics.mean_squared_error(test_target, model.predict(test_data)))
        print("Test RMSE on fold {}: SGD {:.2f}, explicit {:.2f}".format(len(rmses), rmses[-1][-1]["test"], explicit_rmse))

    if args.plot:
        for i in range(len(rmses)):
            plt.plot(range(args.iterations), [x["train"] for x in rmses[i]], label="train-{}".format(i))
            plt.plot(range(args.iterations), [x["test"] for x in rmses[i]], label="test-{}".format(i))

        plt.legend(loc="lower right")
        plt.show()
