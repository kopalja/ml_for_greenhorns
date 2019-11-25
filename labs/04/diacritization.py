#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import sklearn.neural_network


#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b

import numpy as np




class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/1920/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8") as dataset_file:
            self.data = dataset_file.read()


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--window", default=4, type=int, help="Random seed")



class Encoder:
    def __init__(self, text, DIA_TO_NODIA):
        self._min = 1000
        self._max = - self._min
        for ch in text:
            code = ord(ch.lower().translate(DIA_TO_NODIA))
            self._min = min(self._min, code)
            self._max = max(self._max, code) 
        self._size = self._max - self._min + 1

    def encode(self, x):
        x = ord(x)
        if x < self._min or x > self._max:
            raise Exception('Unknown char to encode')
        one_hot = np.zeros(self._size)
        one_hot[x - self._min] = 1
        return one_hot

#########################################################################

index_map = [200, 100, 50, 120, 200, 50, 50, 100, 100, 50, 100, 200, 100 ]



class Net:
    def __init__(self, encoder, index):
        self._encoder = encoder
        self._net = sklearn.neural_network.MLPClassifier(
            hidden_layer_sizes=(index_map[index],),
            activation='relu',
            batch_size=50,
            alpha=0.005,
            max_iter=10,
            verbose=True
        )
        self._inpts = []
        self._targets = []

    def add_sample(self, window, target):
        self._inpts.append(np.concatenate([self._encoder.encode(ch) for ch in window]))
        self._targets.append(ord(target))
        

    def train(self):
        self._output_to_code_map = {}
        tmp = self._targets.copy()
        mins = []
        i = 0
        while min(tmp) < 1000:
            minn = min(tmp)
            mins.append(minn)
            self._output_to_code_map[i] = minn
            tmp = [t if t > minn else 1000 for t in tmp]
            i += 1

        targets = [mins.index(target) for target in self._targets]
        self._net.fit(self._inpts, targets)
        self._inpts = None
        self._targets = None

    def predict(self, window):
        inpt = np.concatenate([self._encoder.encode(ch) for ch in window])
        output = self._net.predict(inpt.reshape(1, -1))
        #print(output)
        return chr(self._output_to_code_map[output[0]])

class Model:
    def __init__(self, dataset, dictionary, args):
        #self._dictionary = dictionary
        self._args = args
        self._encoder = Encoder(dataset.data, dataset.DIA_TO_NODIA)
        self._model = dict()

        self._LETTERS_DIA = dataset.LETTERS_DIA
        self._LETTERS_NODIA = dataset.LETTERS_NODIA
        self._DIA_TO_NODIA = dataset.DIA_TO_NODIA
        index = 0
        for l in dataset.LETTERS_NODIA:
            if l not in self._model.keys():
                self._model[l] = Net(self._encoder, index)
                index += 1

    def _create_window(self, text, index):
        left = list(text[index - self._args.window : index])
        right = list(text[index + 1 : index + self._args.window + 1])

        left = [l.translate(self._DIA_TO_NODIA) for l in left]
        right = [l.translate(self._DIA_TO_NODIA) for l in right]
        left.reverse()
        space = False
        for i, t in enumerate(left):
            if t == ' ':
                space = True
            if space:
                left[i] = ' '
        left.reverse()

        space = False
        for i, t in enumerate(right):
            if t == ' ':
                space = True
            if space:
                right[i] = ' '
        return left + right
        

    def train(self, text):
        #self._count_words(text)
        for index, ch in enumerate(text):
            was_cap = ch.lower() != ch
            ch = ch.lower()

            if (ch in self._LETTERS_DIA or ch in self._LETTERS_NODIA) \
            and index >= self._args.window and index < len(text) - self._args.window:
                window = text[index - self._args.window : index] + text[index + 1 : index + self._args.window + 1]
                #self._model[ch.translate(self._DIA_TO_NODIA)].add_sample(window, ch)
                self._model[ch.translate(self._DIA_TO_NODIA)].add_sample(self._create_window(text, index), ch)

        for _, model in self._model.items():
            model.train()


    def _count_words(self, text):
        self._counts = {}
        text = text.split(' ')
        for i, word  in enumerate(text):
            no_dia = word.translate(self._DIA_TO_NODIA)
            if no_dia in self._dictionary.variants.keys():
                if no_dia not in self._counts:
                    self._counts[no_dia] = np.zeros(len(self._dictionary.variants[no_dia]))

                for i, pos in enumerate(self._dictionary.variants[no_dia]):
                        self._counts[no_dia][i] += 1
                        break
                    # if i == len(self._counts[no_dia]) - 1:
                    #     raise Exception('word doesnt exist')


    def _pick_substitution(self, word, d):
        no_dia = word.translate(self._DIA_TO_NODIA)
        min_error = 123
        result = str()
        for candidate in d.variants[no_dia]:
            error = 0
            for ch1, ch2  in zip(candidate, word):
                if  ch1 != ch2:
                    error += 1
            if error < min_error:
                min_error = error
                result = candidate
        return result

    def _pick_substitution2(self, word):
        no_dia = word.translate(self._DIA_TO_NODIA)
        if no_dia not in self._counts:
            return self._dictionary.variants[no_dia][0]

        index = self._counts[no_dia].argmax()
        return self._dictionary.variants[no_dia][index]


    def predict(self, text, d):
        last_word = str()
        r = []
        for index, ch in enumerate(text):
            was_cap = ch.lower() != ch
            ch = ch.lower()

            # word ended
            if ch == ' ':
                last_no_dia = last_word.translate(self._DIA_TO_NODIA)
                # created unexisting word
                if last_no_dia in d.variants.keys() and last_word not in d.variants[last_no_dia]:
                    #print('found mistake')
                    r = r[:-len(last_word)]
                    new_word = self._pick_substitution(last_word, d)
                    #r.extend(dictionary.variants[last_no_dia][0])
                    r.extend(new_word)
                last_word = str()
                r.append(' ')
            # char with possible dia 
            elif (ch in self._LETTERS_DIA or ch in self._LETTERS_NODIA) \
            and index >= self._args.window and index < len(text) - self._args.window:
                #window = text[index - self._args.window : index] + text[index + 1 : index + self._args.window + 1]
                prediciton = self._model[ch.translate(self._DIA_TO_NODIA)].predict(self._create_window(text, index))
                #prediciton = ch
                if was_cap:
                    prediciton = prediciton.upper()
                last_word += prediciton
                r.append(prediciton)

            else:
                if was_cap:
                    ch = ch.upper()
                last_word += ch
                r.append(ch)
        return ''.join(r)



if __name__ == "__main__":
    from diacritization import Model, Net, Encoder
    from diacritization_dictionary import Dictionary

    dictionary = Dictionary()
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load the dataset, downloading it if required
    train = Dataset()


    number = len(train.data) 
    training_text = train.data[:number]
    testing_text = train.data[number:]

    # TODO: Train the model.
    model = Model(train, dictionary, args)
    model.train(training_text)


    # with lzma.open('diacritization.model', "rb") as model_file:
    #     model = pickle.load(model_file)


    print(len(testing_text))
    result_text = model.predict(testing_text, dictionary)

    with open("gold.txt", "w", encoding="utf-8") as gold:
        gold.write(testing_text)

    with open("system.txt", "w", encoding="utf-8") as system:
        system.write(result_text)



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
    # The `data` is a `str` containing text without diacritics
    #import diacritization
    args = parser.parse_args([])

    # TODO: Predict target values for the given data.
    #
    # You should probably start by loading a model. Start by opening the model
    # file for binary read access and then use `pickle.load` to deserialize the
    # model from the stored binary data:
    with lzma.open(args.model_path, "rb") as model_file:
        model = pickle.load(model_file)

    # TODO: Return the predictions as a diacritized `str`. It has to have
    # exactly the same length as `data`.
    from diacritization_dictionary import Dictionary
    d = Dictionary()
    return model.predict(data, d)
