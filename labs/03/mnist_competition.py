
#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b

import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np
from scipy import ndimage

import sklearn.neural_network
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.pipeline

class Dataset:
    def __init__(self,
                 name="mnist.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/1920/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        dataset = np.load(name)
        self.images = dataset["data"] / 255.0
        self.target = dataset["target"]
    
    def expanded_dataset(self, shift_max=5, shift_min=2, shift_count=2, rotations_max=11, rotations_min=2.5, rotations_count=3, verbose=True):
        target = list(self.target)
        images = list(self.images)

        if shift_count > 0 and shift_max > 0:
            if verbose:
                print('Expanding dataset by adding shifts...')
            for image, y in zip(self.images, self.target):
                shifts = set()
                while len(shifts) < shift_count:
                    shift = tuple(np.random.randint(-shift_max, shift_max + 1, 2))
                    if shift in shifts or abs(shift[0]) + abs(shift[1]) < shift_min:
                        continue
                    if verbose and len(images) % 10_000 == 0:
                        print(f'Dataset size is {len(images)}')
                    shifts.add(shift)
                    images.append(ndimage.shift(image, shift, cval=0))
                    target.append(y)
            if verbose:
                print(f'Finished expanding by shifts, dataset size is {len(images)}')
        im_count_shifts = len(images)
        if rotations_count > 0 and rotations_max > 0:
            if verbose:
                print('Expanding dataset by adding rotations...')
                print(f'Adding {rotations_count} rotations per image')
            for image, y in zip(images[:im_count_shifts], target[:im_count_shifts]):
                for _ in range(rotations_count):
                    angle = np.random.uniform(-rotations_max, rotations_max)
                    if abs(angle) < rotations_min:
                        continue
                    if verbose and len(images) % 10_000 == 0:
                        print(f'Dataset size is {len(images)}')
                    images.append(ndimage.rotate(image, angle, cval=0, reshape=False))
                    target.append(y)
            if verbose:
                print(f'Finished expanding by rotations, dataset size is {len(images)}')

        return np.asarray(images, dtype='float32').reshape([-1, 28*28]), np.asarray(target, dtype='int32')
                

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

if __name__ == "__main__":
    args = parser.parse_args([])

    # Set random seed
    np.random.seed(args.seed)

    # Load the dataset, downloading it if required
    train = Dataset()

    # TODO: Train the model.
    data, target = train.expanded_dataset()

    data_train, data_val, target_train, target_val = sklearn.model_selection.train_test_split(data, target, test_size=0.1)
    print(f'Training on {len(data_train)} images')
    model = sklearn.neural_network.MLPClassifier(
        hidden_layer_sizes=(800,),
        activation='logistic',
        batch_size=50,
        alpha=0.00005,
        max_iter=90,
        verbose=True,
    )

    model.fit(data_train, target_train)
    print(model.score(data_train, target_train))
    print(model.score(data_val, target_val))

    # TODO: The trained model needs to be saved. All sklearn models can
    # be serialized and deserialized using the standard `pickle` module.
    # Additionally, we also compress the model.
    #
    # To save a model, open a target file for binary access, and use
    # `pickle.dump` to save the model to the opened file:
    with lzma.open(args.model_path, "wb") as model_file:
        pickle.dump(model, model_file)

# The `recodex_predict` is called during ReCodEx evaluation (there can be
# several Python sources in the submission, but exactly one should contain
# a `recodex_predict` method).
def recodex_predict(data):
    # The `data` is a numpy arrap containt test set input.

    args = parser.parse_args([])

    # TODO: Predict target values for the given data.
    #
    # You should probably start by loading a model. Start by opening the model
    # file for binary read access and then use `pickle.load` to deserialize the
    # model from the stored binary data:
    with lzma.open(args.model_path, "rb") as model_file:
        model = pickle.load(model_file)

    # TODO: Return the predictions as a Numpy array.
    return model.predict(data / 255.0)