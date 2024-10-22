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
from tensorflow.python.lib.io import file_io
import argparse
import pdb


data_path = "gs://scfi-nn"

def read_chars(filename):
    with tf.gfile.GFile(filename, "r") as f:
        data = f.read()
        chars = list(set(data))
        return data, chars

def build_vocab(filename):
    data, _ = read_chars(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    chars, _ = list(zip(*count_pairs))
    char_to_id = dict(zip(chars, range(len(chars))))

    return char_to_id

def file_to_char_ids(filename, char_to_id):
    data, _ = read_chars(filename)
    return [char_to_id[char] for char in data if char in char_to_id]

def load_data():
    train_path = os.path.join(data_path, "combined.train.txt")
    valid_path = os.path.join(data_path, "combined.valid.txt")

    #build the vocab
    char_to_id = build_vocab(train_path)
    train_data = file_to_char_ids(train_path, char_to_id)
    valid_data = file_to_char_ids(valid_path, char_to_id)
    vocabulary = len(char_to_id)
    reversed_dictionary = dict(zip(char_to_id.values(), char_to_id.keys()))

    return train_data, valid_data, vocabulary, reversed_dictionary

def train_model(job_dir, **args):
    train_data, valid_data, vocabulary, reversed_dictionary = load_data()
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

    num_steps = 140 #30
    batch_size = 30 #20
    train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary, skip_step=num_steps)
    valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary, skip_step=num_steps)

    hidden_size = 700
    use_dropout = True
    # Build the network using sequential
    model = Sequential()
    model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    if use_dropout:
        model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(vocabulary)))
    model.add(Activation('relu'))


    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['categorical_accuracy'])

    print(model.summary())
    num_epochs = 20

    model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs, \
    validation_data=valid_data_generator.generate(), validation_steps=len(valid_data)//(batch_size * num_steps))

    model.save("final_model.hdf5")
    # Save model.h5 on to google storage
    with file_io.FileIO('final_model.hdf5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/final_model.hdf5', mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')

    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    train_model(**arguments)
