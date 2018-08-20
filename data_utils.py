
from __future__ import print_function
import collections
import os
import numpy as np
import tensorflow as tf
import re
import glob

def get_current_model(data_path):
    currentModel = max(glob.glob(data_path + 'model-[0-9]*.hdf5'))
    currentModelNumber = int(re.search(r"model-(\d*)", currentModel).group(1))
    print("Loading Model ", end="")
    print(currentModelNumber)
    return currentModel, currentModelNumber

def read_chars(filename):
    data = open(filename, "r").read()
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

def load_data(data_path):
    # TODO Save char_to_id and reversed_dictionary to file for JSON use
    train_path = os.path.join(data_path, "input_data/combined.train.txt")
    valid_path = os.path.join(data_path, "input_data/combined.valid.txt")

    #build the vocab
    char_to_id = build_vocab(train_path)
    train_data = file_to_char_ids(train_path, char_to_id)
    valid_data = file_to_char_ids(valid_path, char_to_id)
    vocabulary = len(char_to_id)
    reversed_dictionary = dict(zip(char_to_id.values(), char_to_id.keys()))

    return train_data, valid_data, vocabulary, reversed_dictionary, char_to_id