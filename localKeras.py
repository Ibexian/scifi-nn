from __future__ import print_function
import collections
import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda, Bidirectional
from keras.layers import LSTM, BatchNormalization
from keras.optimizers import RMSprop, Adam, SGD, Adamax
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TerminateOnNaN
from data_utils import get_current_model, load_data
import argparse
import pdb

data_path = "/Users/wkamovitch/Sites/scifinn/"

parser = argparse.ArgumentParser()
parser.add_argument('run_opt', type=str, default="train", help='Mode to start the keras model with')
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
args = parser.parse_args()

if args.data_path:
    data_path = args.data_path

train_data, valid_data, vocabulary, reversed_dictionary, _ = load_data(data_path)
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

num_steps = 50 #30
batch_size = 128 #20
train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary, skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary, skip_step=num_steps)

hidden_size = 500
# Build the network using sequential
model = Sequential()
# Convert Indexes to Dense Vectors
model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
# Three Layers of LSTM
model.add(LSTM(hidden_size, return_sequences=True, activation='relu'))
model.add(LSTM(hidden_size, return_sequences=True, activation='relu'))
model.add(LSTM(hidden_size, return_sequences=True, activation='relu'))
# Use normalization and Dropout to prevent overfitting
model.add(BatchNormalization())
model.add(Dropout(0.5))
# Add layer for each time step (good for sequencial data)
model.add(TimeDistributed(Dense(vocabulary)))
model.add(Activation('softmax'))
# Begin by using Adadelta for the first few epochs
model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['categorical_accuracy'])
print(model.summary())
checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
num_epochs = 500

if args.run_opt == "train":
    model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs, \
        validation_data=valid_data_generator.generate(), validation_steps=len(valid_data)//(batch_size * num_steps), callbacks=[checkpointer, TerminateOnNaN()])

    model.save(data_path + "final_model.hdf5")
elif args.run_opt == "continue":
    currentModel, currentModelNumber = get_current_model(data_path)
    model = load_model(currentModel)
    bigger_batch_size = batch_size + (currentModelNumber * 20)
    batch_size = bigger_batch_size if bigger_batch_size < 2600 else 2600 #Assign a new batch size when continuing training (this only fires when starting/restarting the training)
    checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1, monitor='categorical_accuracy', save_best_only=True, mode='auto')
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.003, momentum=0.000003), metrics=['categorical_accuracy'])
    print("Learning Rate: ", end="")
    print(K.eval(model.optimizer.lr))
    print("Batch Size: ", end="")
    print(batch_size)
    model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs, \
        validation_data=valid_data_generator.generate(), validation_steps=len(valid_data)//(batch_size * num_steps), callbacks=[checkpointer, TerminateOnNaN()], initial_epoch=currentModelNumber)

    model.save(data_path + "final_model.hdf5")
elif args.run_opt == "test":
    currentModel, _ = get_current_model(data_path)
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
