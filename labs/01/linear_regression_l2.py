#!/usr/bin/env python3


#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b

import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", default=True, action="store_true", help="Plot the results")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=50, type=int, help="Test size to use")
    args = parser.parse_args()

    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()

    # TODO: Split the dataset randomly to train and test using
    # `sklearn.model_selection.train_test_split`, with
    # `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = \
         sklearn.model_selection.train_test_split(dataset.data, dataset.target, test_size = args.test_size, random_state = args.seed)


    # TODO: Using sklearn.linear_model.Ridge, fit the train set using
    # L2 regularization, employing lambdas from 0 to 100 with a step size 0.1
    # (note that lambda is actually called `alpha` in Ridge constructor).
    #
    # For every model, compute the root mean squared error, and print out the
    # lambda producing lowest test error and the test error itself.
    # (The `sklearn.metrics.mean_squared_error` may come handy to compute at
    # least mean quared error; root mean squared error itself is not provided
    # by sklear.metrics.)
    lambdas = []
    rmses = []
    best_lambda = None
    best_rmse = None
    for i in range(0, 1000):
        l = i / 10
        ridge = sklearn.linear_model.Ridge(l)
        ridge.fit(train_data, train_target)
        predictions = ridge.predict(test_data)
        mse = sklearn.metrics.mean_squared_error(predictions, test_target)
        rmse = (mse)**(1/2)
        rmses.append(rmse)
        lambdas.append(l)
        if best_lambda is None or best_rmse > rmse:
            best_rmse = rmse
            best_lambda = l

    with open("linear_regression_l2.out", "w") as output_file:
        print("{:.1f}, {:.2f}".format(best_lambda, best_rmse), file=output_file)

    if args.plot:
        # TODO: This block is not part of ReCodEx submission, so you
        # will get points even without it. However, it is useful to
        # learn to visualize the results.

        # If you collect used lambdas to `lambdas` and their respective
        # results to `rmses`, the following lines will plot the result
        # if you add `--plot` argument.
        import matplotlib.pyplot as plt
        plt.plot(lambdas, rmses)
        plt.show()
