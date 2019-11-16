#!/usr/bin/env python3

#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b

import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_ratio", default=0.5, type=float, help="Test set size ratio")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load digit dataset
    dataset = sklearn.datasets.load_digits()
    print(dataset.DESCR, file=sys.stderr)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_ratio` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target, test_size=args.test_ratio, random_state=args.seed)

    # TODO: Create a pipeline, which
    # 1. performs sklearn.preprocessing.MinMaxScaler()
    # 2. performs sklearn.preprocessing.PolynomialFeatures()
    # 3. performs sklearn.linear_model.LogisticRegression(multi_class="multinomial")
    #
    # Then, using sklearn.model_selection.StratifiedKFold(5), evaluate crossvalidated
    # train performance of all combinations of the the following parameters:
    # - polynomial degree: 1, 2
    # - LogisticRegression regularization C: 0.01, 1, 100
    # - LogisticRegression solver: lbgfs, sag
    #
    # For the best combination of parameters, compute the test set accuracy.
    #
    # The easiest way is to use `sklearn.model_selection.GridSearchCV`.

    pipeline = sklearn.pipeline.Pipeline(steps = [
        ('MinMaxScaler', sklearn.preprocessing.MinMaxScaler()),
        ('Polynomial', sklearn.preprocessing.PolynomialFeatures()),
        ('Regression', sklearn.linear_model.LogisticRegression(multi_class="multinomial"))
    ])

    #print(pipeline.get_params().keys()
    parameters = { 
        'Polynomial__degree' : [1, 2], 
        'Regression__C': [0.01, 1, 100],
        'Regression__solver': ['sag', 'lbfgs']
    }
    grid = sklearn.model_selection.GridSearchCV(pipeline, parameters, scoring='accuracy', refit=True, cv=sklearn.model_selection.StratifiedKFold(5))
    grid.fit(train_data, train_target)
    test_accuracy = grid.score(test_data, test_target)
    print("{:.2f}".format(100 * test_accuracy))