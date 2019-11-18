import urllib.request
import os
import sys
import pandas as pd

class Dataset:
    def __init__(self,
                 name="binary_classification_competition.train.csv.xz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/1920/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `train_target` (column Target)
        # and `train_data` (all other columns).
        dataset = pd.read_csv(name)
        self.data, self.target = dataset.drop("Target", axis=1), dataset["Target"]