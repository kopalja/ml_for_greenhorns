#!/usr/bin/env python3

import pickle
from diacritization import Model, Net, Encoder

def f():
    with open('diacritization.model', "rb") as model_file:
        model = pickle.load(model_file)
    print('Load ok')



if __name__ == "__main__":
    print('main')
    f()
else:
    print('import')
    f()


