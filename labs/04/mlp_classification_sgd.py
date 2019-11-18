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

def relu(x):
    return np.maximum(0, x)

def softmax(z):
    out = np.exp(z - np.max(z))
    return out / np.sum(out)

def one_hot_encoding(x, classes):
    one_hot = np.zeros(classes)
    one_hot[x] = 1
    return one_hot

def d_relu(d_init, out):
    d = np.array(d_init, copy = True)
    d[out < 0] = 0.
    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
    parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
    parser.add_argument("--hidden_layer", default=20, type=int, help="Hidden layer size")
    parser.add_argument("--iterations", default=50, type=int, help="Number of iterations over the data")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=797, type=int, help="Test set size")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data
    data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights
    weights = [np.random.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               np.random.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]

    def forward(inputs):
        # TODO: Implement forward propagation, returning *both* the value of the hidden
        # layer and the value of the output layer.
        #
        # We assume a neural network with a single hidden layer of size `args.hidden_layer`
        # and ReLU activation, where ReLU(x) = max(x, 0), and an output layer with softmax
        # activation.
        #
        # The value of the hidden layer is computed as ReLU(inputs times weights[0]).
        # The value of the output layer is computed as softmax(hidden_layer times weights[1]).
        #
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you can use the fact that
        # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
        # That way we only exponentiate values which are non-positive, and overflow does not occur.
        y_hidden = relu(inputs @ weights[0])
        y_out = softmax(y_hidden @ weights[1])

        return y_hidden, y_out

    for iteration in range(args.iterations):
        train_correct = 0
        permutation = np.random.permutation(train_data.shape[0])
        grad = [np.zeros_like(weights[0]), np.zeros_like(weights[1])]
        for i, perm in enumerate(permutation):
            inp, target = train_data[perm], train_target[perm]
            y_hidden, y_out = forward(inp)
            label = one_hot_encoding(target, args.classes)
            delta = label - y_out
            g = y_hidden.reshape(-1, 1) @ delta.reshape(1, -1)
            grad[1] += g
            delta = (delta @ weights[1].T)
            g = (inp.reshape(-1, 1) @ delta.reshape(1, -1)) * (y_hidden > 0)
            grad[0] += g
            if (i + 1) % args.batch_size == 0:
                weights[0] += args.learning_rate * grad[0] / args.batch_size
                weights[1] += args.learning_rate * grad[1] / args.batch_size
                grad = [np.zeros_like(weights[0]), np.zeros_like(weights[1])]
                
                

        # TODO: Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # The gradient used in SGD has now two parts, gradient of weghts[0] and weights[1].
        #
        # You can either compute the gradient directly from the neural network formula,
        # i.e., as a gradient of -log P(target | data), or you can compute
        # it "step by step" using chain rule of derivatives, in the following order:
        # - compute the derivative of the loss with respect to *inputs* of the
        #   softmax on the last layer (we did this already in softmax_classification_sgd)
        # - compute the derivative with respect to weights[1]
        # - compute the derivative with respect to the hidden layer output
        # - compute the derivative with respect to the hidden layer input
        # - compute the derivative with respect to weights[0]

        # TODO: After the SGD iteration, measure the accuracy for both the
        # train test and the test set and print it in percentages.
        predictions = np.asarray([forward(x)[1] for x in test_data])
        test_acc = np.mean(predictions.argmax(axis=1) == test_target)
        predictions = np.asarray([forward(x)[1] for x in train_data])
        train_acc = np.mean(predictions.argmax(axis=1) == train_target)
        print("After iteration {}: train acc {:.1f}%, test acc {:.1f}%".format(
            iteration + 1,
            100 * train_acc,
            100 * test_acc,
        ))