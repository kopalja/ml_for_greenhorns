#!/usr/bin/env python3


#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b


import argparse

import numpy as np
import sklearn.datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", default=50, type=int, help="Test size to use")
    args = parser.parse_args()

    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()
    print(dataset.DESCR)



    # The input data are in dataset.data, targets are in dataset.target.
    # TODO: Pad a value of `1` to every instance in dataset.data
    # (np.pad or np.concatenate might be useful).
    dataset.data = np.concatenate((dataset.data, np.ones(shape = (dataset.data.shape[0], 1))), axis = 1)

    # TODO: Split data so that the last `args.test_size` data are the test
    # set and the rest is the training set.
    test_data = {
        "data": dataset.data[-args.test_size:],
        "target": dataset.target[-args.test_size:]
    }
    training_data = {
      "data": dataset.data[:-args.test_size],
      "target": dataset.target[:-args.test_size]  
    }

    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using np.linalg.inv).
    inv = np.linalg.inv(np.matmul(np.matrix.transpose(training_data["data"]), training_data["data"]))
    w = np.matmul(np.matmul(inv, np.matrix.transpose(training_data["data"])),  training_data["target"])

    # TODO: Predict target values on the test set.
    predicitons = np.matmul(test_data["data"], w)

    # TODO: Compute root mean square error on the test set predictions.
    rmse = np.sqrt(np.mean((predicitons - test_data["target"])**2))

    with open("linear_regression_manual.out", "w") as output_file:
        print("{:.2f}".format(rmse), file=output_file)
