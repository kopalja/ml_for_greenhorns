#!/usr/bin/env python3

#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection


def model(weights, inpt):
    return 1 / (1 + np.exp(- np.dot(weights, inpt)))

def CrossEntropy(prediction, y):
    if y == 1:
      return -np.log(prediction)
    else:
      return -np.log(1 - prediction)

def batch_update(features, labels, weights, lr):
    predictions = np.array([model(weights, f) for f in features])
    gradient = np.dot(np.matrix.transpose(features),  predictions - labels)
    return weights - lr * gradient / len(features)

def mean(x):
    return sum(x) / len(x)

def loss_acc(features, targets):
    loss = [CrossEntropy(model(weights, feature), target)for feature, target in zip(features, targets)]
    acc = [1 if round(model(weights, feature)) == target else 0 for feature, target in zip(features, targets)]
    return mean(loss), mean(acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
    parser.add_argument("--examples", default=200, type=int, help="Number of examples")
    parser.add_argument("--iterations", default=50, type=int, help="Number of iterations over the data")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--plot", default=False, action="store_true", help="Plot progress")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_ratio", default=0.5, type=float, help="Test set size ratio")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Generate an artifical regression dataset
    data, target = sklearn.datasets.make_classification(
        n_samples=args.examples, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)

    # Append a constant feature with value 1 to the end of every input data
    data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_ratio` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_ratio, random_state=args.seed)



    # Generate initial linear regression weights
    weights = np.random.uniform(size=train_data.shape[1])

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

        # TODO: For every episode, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or crossentropy loss, or KL loss) per example.
        # Print the accuracies in percentages.
        train_loss, train_acc = loss_acc(train_data, train_target)
        test_loss, test_acc = loss_acc(test_data, test_target)
        print("After iteration {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            iteration + 1,
            train_loss,  # The average train loss,
            100 * train_acc,  # Training accuracy,
            test_loss, # The average test loss,
            100 * test_acc # Test accuracy,
        ))

        if args.plot:
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 20)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 20)
            predictions = [[1 / (1 + np.exp(-([x, y, 1] @ weights))) for x in xs] for y in ys]
            plt.contourf(xs, ys, predictions, levels=40, cmap=plt.cm.RdBu, alpha=0.7)
            plt.contour(xs, ys, predictions, levels=[0.25, 0.5, 0.75], colors="k")
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="P", label="train", cmap=plt.cm.RdBu)
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, label="test", cmap=plt.cm.RdBu)
            plt.legend(loc="upper right")
            plt.show()
