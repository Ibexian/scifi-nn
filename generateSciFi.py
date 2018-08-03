from data_utils import *
from keras.models import load_model
import numpy as np
import argparse

data_path = "/Users/wkamovitch/Sites/scifinn/"
model_number = "04"
num_predict = 140
num_steps = 140

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
parser.add_argument('--length', type=int, default=140, help='Length of generated output')
parser.add_argument('--model', type=str, default=model_number, help='Model number or name of desired model - in the model-{number} format')
args = parser.parse_args()

if args.data_path:
    data_path = args.data_path

if args.length:
    num_predict = args.length

if args.model:
    model_number = args.model

#Load in the text data
_, _, vocabulary, reversed_dictionary = load_data(data_path)

model = load_model(data_path + "model-" + model_number + ".hdf5")

def generate_text(model, input_length, output_length, vocab_size, reversed_dictionary):
    print("Generated Text: ")
    y_char = []
    X = np.random.random_integers(low=0, high=vocab_size - 1, size=(1, input_length))
    for i in range(output_length):
        # Make a prediction
        prediction = model.predict(X)
        ix = np.argmax(prediction[:, input_length - 1, :])
        # Assign that prediction as the last val in X
        X[0][0] = ix
        X = np.roll(X, -1, axis=1)
        # Add Predition to output
        print(reversed_dictionary[ix], end="")
        y_char.append(reversed_dictionary[ix])
    return('').join(y_char)

generate_text(model, num_steps, num_predict, vocabulary, reversed_dictionary)
print("")