#!/usr/bin/env python3
#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.neighbors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", default=5, type=int, help="Number of clusters")
    parser.add_argument("--examples", default=140, type=int, help="Number of examples")
    parser.add_argument("--iterations", default=6, type=int, help="Number of kmeans iterations to perfom")
    parser.add_argument("--plot", default=False, action="store_true", help="Plot the results")
    parser.add_argument("--seed", default=44, type=int, help="Random seed")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Generate artificial data
    data, target = sklearn.datasets.make_blobs(n_samples=args.examples, centers=args.clusters, n_features=2, random_state=args.seed)

    # Start by using the first `args.clusters` samples as the cluster representations.
    centers = data[:args.clusters].copy()
    model = sklearn.neighbors.KNeighborsClassifier(1)

    # TODO: Run `args.iterations` of the kmeans algorithm, storing zero-based cluster assignment
    # to `clusters`.
    for it in range(args.iterations):
        model.fit(centers, list(range(args.clusters)))
        clusters = model.predict(data)
        centers = np.array([np.mean(data[clusters == i], axis=0) for i in range(args.clusters)])

    if args.plot:
        plt.gca().set_aspect(1)
        plt.scatter(data[:, 0], data[:, 1], c=clusters)
        plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=200, c="#ff0000")
        plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=50, c=range(args.clusters))
        plt.show()

    # Print the zero-based cluster assignments, one input example per line
    for cluster in clusters:
        print(cluster)
