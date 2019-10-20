#!/usr/bin/env python3

#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b

import argparse
import pickle

import numpy as np
import sklearn.linear_model

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="linear_regression_competition.model", type=str, help="Model path")
parser.add_argument("--seed", default=42, type=int, help="Random seed")





if __name__ == "__main__":
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load the data to train["data"] and train["target"]
    dataset = np.load("linear_regression_competition.train.npz")
    dataset = {entry: dataset[entry] for entry in dataset}

    real_features = np.delete(np.copy(dataset["data"]), [0, 1, 2, 3, 4, 5, 6, 7], 1)

    n2 = np.copy(dataset["data"])
    n2 = np.power(n2, 2)
    n3 = np.copy(dataset["data"])
    n3 = np.power(n3, 3)
    n4 = np.copy(real_features)
    n4 = np.power(n4, 4)
    dataset["data"] = np.concatenate((dataset["data"], n2, n3), axis = 1)

    # split dataset
    test_size = 50
    train = {"data": None, "target": None}
    test = {"data": None, "target": None}


    # rmses = []
    # for i in range(100):
    #     train["data"], test["data"], train["target"], test["target"] = sklearn.model_selection.train_test_split(dataset["data"], dataset["target"], test_size = test_size, random_state = i)

    #     # TODO: Train the model
    #     #alfa = find_best_alfa(train, test)

    #     #model = sklearn.linear_model.Ridge(alpha=0.5, tol=0.00001, max_iter=30000, normalize=True)
    #     model = sklearn.linear_model.LinearRegression()
    #     #model = sklearn.linear_model.ElasticNet(alpha = 0.05, l1_ratio=0.7,  max_iter=20000)
    #     #model = sklearn.linear_model.TheilSenRegressor()
    #     #model = sklearn.linear_model.Lasso(alpha=0.5, tol=0.00001, max_iter=30000)

    #     model.fit(train["data"], train["target"])

    #     predictions = model.predict(test["data"])

    #     predictions = np.array([1 if p < 1 else round(p) for p in predictions])

    #     mse = sklearn.metrics.mean_squared_error(predictions, test["target"])
    #     rmse = (mse)**(1/2)
    #     rmses.append(rmse)
    # print("output {0}".format(np.mean(rmses)))

    model = sklearn.linear_model.LinearRegression()
    model.fit(dataset["data"], dataset["target"])



    # TODO: The trained model needs to be saved. All sklear models can
    # be serialized and deserialized using the standard `pickle` module.
    #
    # To save a model, open a target file for binary access, and use
    # `pickle.dump` to save the model to the opened file:
    with open(args.model_path, "wb") as model_file:
        pickle.dump(model, model_file)



# The `recodex_predict` is called during ReCodEx evaluation (there can be
# several Python sources in the submission, but exactly one should contain
# a `recodex_predict` method).
def recodex_predict(data):
    # The `data` is a Numpy array containt test set input.

    args = parser.parse_args([])

    # TODO: Predict target values for the given data.
    #
    # You should probably start by loading a model. Start by opening the model
    # file for binary read access and then use `pickle.load` to deserialize the
    # model from the stored binary data:
    with open(args.model_path, "rb") as model_file:
        model = pickle.load(model_file)


    n2 = np.copy(data)
    n2 = np.power(n2, 2)
    n3 = np.copy(data)
    n3 = np.power(n3, 3)
    data = np.concatenate((data, n2, n3), axis = 1)
    # TODO: Return the predictions as a Numpy array.
    predictions = model.predict(data)
    return np.array([1 if p < 1 else round(p) for p in predictions])


