#!/usr/bin/env python3
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b


def softmax(x):
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum(axis=0) 


def one_hot_encoding(x, classes):
    one_hot = np.zeros(classes)
    one_hot[x] = 1
    return one_hot


def model(weights, inpt):
    return softmax(np.dot(weights.T, inpt))
    #return 1 / (1 + np.exp(- np.dot(weights, inpt)))

def batch_update(inpt, labels, weights, lr):
    predictions = np.array([model(weights, f) for f in inpt])
    labels = np.array([one_hot_encoding(label, args.classes) for label in labels])
    gradient = np.dot(np.matrix.transpose(inpt),  predictions - labels)
    return weights - lr * gradient / inpt.shape[0]


def mean(x):
    return sum(x) / len(x)

def acc(weights, features, targets):
    ac = [1 if model(weights, feature).argmax() == target else 0 for feature, target in zip(features, targets)]
    return mean(ac)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int, help="Batch size")
    parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
    parser.add_argument("--iterations", default=100, type=int, help="Number of iterations over the data")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=797, type=int, help="Test set size")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # exit()

    # Append a constant feature with value 1 to the end of every input data
    data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)


    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)


    # Generate initial model weights
    weights = np.random.uniform(size=[train_data.shape[1], args.classes])

    for iteration in range(args.iterations):
        permutation = np.random.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        for i in range(train_data.shape[0] // args.batch_size):
            index = i * args.batch_size
            weights = batch_update(
                train_data[permutation[index:index+args.batch_size]], 
                train_target[permutation[index:index+args.batch_size]], 
                weights, 
                args.learning_rate
            )

        # TODO: After the SGD iteration, measure the accuracy for both the
        # train test and the test set and print it in percentages.
        train_acc = acc(weights, train_data, train_target)
        test_acc = acc(weights, test_data, test_target)
        print("After iteration {}: train acc {:.1f}%, test acc {:.1f}%".format(
            iteration + 1,
            100 * train_acc,
            100 * test_acc
        ))
