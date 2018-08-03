from __future__ import print_function
import collections
import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from data_utils import *
import argparse
import pdb

data_path = "/Users/wkamovitch/Sites/scifinn/"

parser = argparse.ArgumentParser()
parser.add_argument('run_opt', type=str, default="train", help='Mode to start the keras model with')
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
args = parser.parse_args()

if args.data_path:
    data_path = args.data_path

train_data, valid_data, vocabulary, reversed_dictionary = load_data(data_path)
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

hidden_size = 500 # Maybe switch to 700
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

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['categorical_accuracy'])

print(model.summary())
checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
num_epochs = 200

if args.run_opt == "train":
    model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs, \
        validation_data=valid_data_generator.generate(), validation_steps=len(valid_data)//(batch_size * num_steps), callbacks=[checkpointer])

    model.save(data_path + "final_model.hdf5")
elif args.run_opt == "continue":
    currentModel, currentModelNumber = get_current_model()
    model = load_model(currentModel)
    model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs, \
        validation_data=valid_data_generator.generate(), validation_steps=len(valid_data)//(batch_size * num_steps), callbacks=[checkpointer], initial_epoch=currentModelNumber)

    model.save(data_path + "final_model.hdf5")
elif args.run_opt == "test":
    currentModel, _ = get_current_model()
    model = load_model(currentModel)
    dummy_iters = 20
    example_training_generator = KerasBatchGenerator(train_data, num_steps, 1, vocabulary, skip_step=1)
    print("Training Data:")
    for i in range(dummy_iters):
        dummy = next(example_training_generator.generate())
    num_predict = 140
    true_print_out = "Actual Words: "
    pred_print_out = "Predicted Words: "
    for i in range(num_predict):
        data = next(example_training_generator.generate())
        prediction = model.predict(data[0])
        predict_word = np.argmax(prediction[:, num_steps-1, :])
        true_print_out += reversed_dictionary[train_data[num_steps + dummy_iters + i]] + ""
        pred_print_out += reversed_dictionary[predict_word] + ""
    print(true_print_out)
    print(pred_print_out)
