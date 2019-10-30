import pandas as pd
import lzma
import argparse
import pickle
import numpy as np

def prepare_data(data, transformer):
    return transformer.transform(data)

def recodex_predict(data):
    # The `data` is a pandas.DataFrame containt test set input.
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path_svm", default="binary_classification_competition-svm.model", type=str, help="SVM model path")
    parser.add_argument("--model_path_rf", default="binary_classification_competition-rf.model", type=str, help="RF model path")
    parser.add_argument("--model_path_et", default="binary_classification_competition-et.model", type=str, help="ET model path")
    parser.add_argument("--model_path_lr", default="binary_classification_competition-lr.model", type=str, help="LR model path")

    parser.add_argument("--model_path_pre", default="binary_classification_competition-pre.model", type=str, help="Preprocessing model path")

    args = parser.parse_args([])
    
    with lzma.open(args.model_path_pre, "rb") as model_file:
        model_pre = pickle.load(model_file)

    data = prepare_data(data.drop('Education', axis=1).to_numpy(), model_pre)

    # TODO: Predict target values for the given data.
    #
    # You should probably start by loading a model. Start by opening the model
    # file for binary read access and then use `pickle.load` to deserialize the
    # model from the stored binary data:

    with lzma.open(args.model_path_rf, "rb") as model_file:
        model_rf = pickle.load(model_file)

    return model_rf.predict(data)

if __name__=='__main__':
    from binary_classification_competition import Dataset
    train = Dataset()
    print(np.mean(recodex_predict(train.data) == train.target))