from __future__ import print_function
import collections
import os
import numpy as np
import tensorflow as tf
import re
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import argparse
import pdb

data_path = "/Users/wkamovitch/Sites/scifinn/"

parser = argparse.ArgumentParser()
parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
args = parser.parse_args()

if args.data_path:
    data_path = args.data_path

def read_words(filename):
    punct = re.compile(r'(\.|\,|\?|\--|\|\)|\(|\!|\;|\_)')
    with tf.gfile.GFile(filename, "r") as f:
        return punct.sub(r' \1 ', f.read().replace("\n", "<eos>")).split()


def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def load_data():
    train_path = os.path.join(data_path, "combined.train.txt")
    valid_path = os.path.join(data_path, "combined.valid.txt")
    test_path = os.path.join(data_path, "combined.test.txt")

    #build the vocab
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    return train_data, valid_data, test_data, vocabulary, reversed_dictionary

train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()
#batch process for input data
class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # track progress of batches in dataset
        self.current_idx = 0
        self.skip_step = skip_step
    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index to zero when at end of data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1: self.current_idx + self.num_steps + 1]
                # convert temp_y to one hot
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y

num_steps = 30 #30
batch_size = 20 #20
train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary, skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary, skip_step=num_steps)

hidden_size = 500
use_dropout = True
# Build the network using sequential
model = Sequential()
model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
if use_dropout:
    model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocabulary)))
model.add(Activation('softmax'))


optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

print(model.summary())
checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
num_epochs = 50

if args.run_opt == 1:
    model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs, \
    validation_data=valid_data_generator.generate(), validation_steps=len(valid_data)//(batch_size * num_steps), callbacks=[checkpointer])

    model.save(data_path + "final_model.hdf5")
elif args.run_opt == 2:
    model = load_model(data_path + "\model-40.hdf5")
    dummy_iters = 40
    example_training_generator = KerasBatchGenerator(train_data, num_steps, 1, vocabulary, skip_step=1)
    print("Training Data:")
    for i in range(dummy_iters):
        dummy = next(example_training_generator.generate())
    num_predict = 10
    true_print_out = "Actual Words: "
    pred_print_out = "Predicted Words: "
    for i in range(num_predict):
        data = next(example_training_generator.generate())
        prediction = model.predict(data[0])
        predict_word = np.argmax(prediction[:, num_steps-1, :])
        true_print_out += reversed_dictionary[train_data[num_steps + dummy_iters + 1]] + " "
        pred_print_out += reversed_dictionary[predict_word] + " "
    print(true_print_out)
    print(pred_print_out)
    # test data set
    dummy_iters = 40
    example_test_generator = KerasBatchGenerator(test_data, num_steps, 1, vocabulary, skip_step=1)
    print("Test Data: ")
    for i in range(dummy_iters):
        dummy = next(example_test_generator.generate())
    num_predict = 10
    true_print_out = "Actual Words: "
    pred_print_out = "Predicted Words: "
    for i in range(num_predict):
        data = next(example_test_generator.generate())
        prediction = model.predict(data[0])
        predict_word = np.argmax(prediction[:, num_steps - 1, :])
        true_print_out += reversed_dictionary[test_data[num_steps + dummy_iters + i]] + " "
        pred_print_out += reversed_dictionary[predict_word] + " "
    print(true_print_out)
    print(pred_print_out)