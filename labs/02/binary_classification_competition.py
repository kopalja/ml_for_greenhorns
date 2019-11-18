#!/usr/bin/env python3

#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b

import argparse
import lzma
import pickle
import os

import numpy as np
import pandas as pd

import sklearn.preprocessing
import sklearn.linear_model
import sklearn.ensemble
from sklearn.compose import ColumnTransformer

from binary_classification_dataset import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model_path_lr", default="binary_classification_competition-lr.model", type=str, help="Logistic regregression model path")
parser.add_argument("--model_path_lr_pca", default="binary_classification_competition-lr-pca.model", type=str, help="LR w/ PCA model path")
parser.add_argument("--model_path_svm", default="binary_classification_competition-svm.model", type=str, help="SVM model path")
parser.add_argument("--model_path_rf", default="binary_classification_competition-rf.model", type=str, help="Random forest model path")
parser.add_argument("--model_path_vote", default="binary_classification_competition-vote.model", type=str, help="Voting model path")
parser.add_argument("--model_path_et", default="binary_classification_competition-et.model", type=str, help="ExtraTrees model path")
parser.add_argument("--model_path_pre", default="binary_classification_competition-pre.model", type=str, help="Preprocessing model path")

parser.add_argument("--seed", default=42, type=int, help="Random seed")

def get_transformer(data):
    # Education provides the same information as Education-num so we drop one of them
    onehot_features = ['Workclass', 'Marital-status', 'Occupation', 'Race', 'Native-country', 'Sex', 'Relationship']
    
    transformers = []
    for (i, name) in enumerate(data.columns):
        if name in onehot_features:
            transformer = sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
        else:
            transformer = sklearn.preprocessing.StandardScaler()
        transformers.append((name, transformer, [i]))

    return ColumnTransformer(transformers)

def fit_svm(data, target):
    svm = sklearn.svm.LinearSVC(dual=False, max_iter=20000)
    parameters_svm = {
        'penalty': ['l1', 'l2'],
        'C': [0.1, 0.3, 1, 30, 100, 300],
    }

    grid_svm = sklearn.model_selection.GridSearchCV(svm, parameters_svm, n_jobs=-1, cv=5)
    grid_svm.fit(data, target)
    print(grid_svm.best_score_)
    print(grid_svm.best_params_)

    with lzma.open(args.model_path_svm, "wb") as model_file:
        pickle.dump(grid_svm.best_estimator_, model_file)

def fit_lr_pca(data, target):
    pca = sklearn.decomposition.PCA(0.99)
    data_pca = pd.DataFrame(data=pca.fit_transform(train.data))

    lr = sklearn.linear_model.LogisticRegression(solver='liblinear')

    d_lr = 1

    parameters_lr_pca = {
        'penalty': ['l1', 'l2'],
        'C': [0.4, 0.8, 1.6, 3.2],
    }

    grid_lr = sklearn.model_selection.GridSearchCV(lr, parameters_lr_pca, n_jobs=-1, cv=5)
    grid_lr.fit(data_pca.iloc[::d_lr], train.target[::d_lr])
    print(grid_lr.best_score_)
    print(grid_lr.best_params_)

def fit_lr(data, target):
    lr = sklearn.linear_model.LogisticRegression()
    
    parameters_lr = {
        'penalty': ['l1', 'l2'],
        'C': [0.45, 0.7, 1, 1.5, 2.25, 3],
    }

    grid_lr = sklearn.model_selection.GridSearchCV(lr, parameters_lr, n_jobs=-1, cv=5)
    grid_lr.fit(train.data.iloc[::d_lr], train.target[::d_lr])
    print(grid_lr.best_score_)
    print(grid_lr.best_params_)

    with lzma.open(args.model_path_lr, "wb") as model_file:
        pickle.dump(grid_lr.best_estimator_, model_file)

def fit_rf(data, target):
    rf = sklearn.ensemble.RandomForestClassifier(501)

    parameters_rf = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [11, 14, 17],
    }

    grid_rf = sklearn.model_selection.GridSearchCV(rf, parameters_rf, refit=True, n_jobs = -1, cv=5)
    grid_rf.fit(train.data, train.target)
    print(grid_rf.best_score_)
    print(grid_rf.best_params_)

    with lzma.open(args.model_path_rf, "wb") as model_file:
        pickle.dump(grid_rf.best_estimator_, model_file)

def fit_et(data, target):
    et = sklearn.ensemble.ExtraTreesClassifier(501)
    parameters_et = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [11, 14, 17],
    }

    grid_et = sklearn.model_selection.GridSearchCV(et, parameters_et, refit=True, n_jobs = -1, cv=5)
    grid_et.fit(train.data, train.target)
    print(grid_et.best_score_)
    print(grid_et.best_params_)

    with lzma.open(args.model_path_et, "wb") as model_file:
        pickle.dump(grid_et.best_estimator_, model_file)

if __name__ == "__main__":
    args = parser.parse_args([])

    # Set random seed
    np.random.seed(args.seed)

    # Load the dataset, downloading it if required
    train = Dataset()
    train.data.drop('Education', axis=1, inplace=True)
    transformer = get_transformer(train.data)

    train.data = transformer.fit_transform(train.data)
    with lzma.open(args.model_path_pre, "wb") as model_file:
        pickle.dump(transformer, model_file)

    #fit_rf(train.data, train.target)
