
#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.compose import ColumnTransformer

#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b




def is_int_column(column):
    for cell in column:
        if not (cell).is_integer():
            return False
    return True


def change_features_order(inpt):
    order = []
    for i in range(inpt.shape[1]):
        if is_int_column(inpt[:, i]):
            order.append(i)
    for i in range(inpt.shape[1]):
        if is_int_column(inpt[:, i]) == False:
            order.append(i)
    return inpt[:, np.array(order)]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="boston", type=str, help="Standard sklearn dataset to load")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_ratio", default=0.5, type=float, help="Test set size ratio")
    args = parser.parse_args()

    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()




    # TODO(linear_regression_l2): Split the dataset randomly to train
    # and test using `sklearn.model_selection.train_test_split`, with
    # `test_size=args.test_ratio` and `random_state=args.seed`.
    train_data, test_data = \
         sklearn.model_selection.train_test_split(dataset["data"], test_size = args.test_ratio, random_state = args.seed)


    # TODO: Process the input columns in the following way:
    # - if a column has only integer values, consider it a categorical column
    #   (days in a week, dog breed, ...; in general integer values can also
    #   represent numerical non-categorical values, but we use this assumption
    #   for the sake of an exercise). Encode the values using one-hot encoding,
    #   probably using `sklearn.preprocess.OneHotEncoder` (note that its output
    #   is by default sparse; you can use `sparse=False` to generate dense output)
    # - for the rest of the columns, normalize their values so that they
    #   have mean 0 and variance 1; you can use `sklearn.preprocessing.StandardScaler`.
    # In the output, there should be first all the one-hot categorical features,
    # and then the real-valued features. To process different dataset columns
    # differently, you can use `sklearn.compose.ColumnTransformer`.

    train_data = change_features_order(train_data)
    test_data = change_features_order(test_data)

    # define column transformers
    transformers = [] 
    for i in range(train_data.shape[1]):
        if is_int_column(train_data[:, i]):
            transformer = sklearn.preprocessing.OneHotEncoder(sparse=False)
        else:
            transformer = sklearn.preprocessing.StandardScaler()
        transformers.append(("column {0}".format(i), transformer, [i]))


    pipeline = sklearn.pipeline.Pipeline(steps = [
        ('columns', ColumnTransformer(transformers)), 
        ('polynomial', sklearn.preprocessing.PolynomialFeatures(2, include_bias=False))
    ])

    train_data = pipeline.fit_transform(train_data)
    test_data = pipeline.transform(test_data)




    # TODO: Generate polynomial features of order 2 from the current features.
    # If the input values are [a, b, c, d], you should generate
    # [a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2]. You can generate such polynomial
    # features either manually, or using
    # `sklearn.preprocess.PolynomialFeatures(2, include_bias=False)`.
    # train_data = pol.fit_transform(train_output)
    # test_data = pol.transform(test_output)

    # TODO: You can wrap all the feature processing steps into one transformer
    # by using `sklearn.pipeline.Pipeline`. Although not strictly needed, it is
    # usually comfortable.

    # TODO: Fit the feature processing steps on the training data.
    # Then transform the training data into `train_data` (you can do both these
    # steps using `fit_transform`), and transform testing data to `test_data`.

    with open("feature_engineering.out", "w") as output_file:
        for data in [train_data, test_data]:
            for line in range(5):
                print(" ".join("{:.6g}".format(data[line, column]) for column in range(data.shape[1])), file=output_file)

                
