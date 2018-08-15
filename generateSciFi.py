from data_utils import load_data
from keras.models import load_model
import numpy as np
import argparse

data_path = "/Users/wkamovitch/Sites/scifinn/"
model_number = "04"
num_predict = 140
num_steps = 50
seed_string = ''

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
parser.add_argument('--length', type=int, default=140, help='Length of generated output')
parser.add_argument('--model', type=str, default=model_number, help='Model number or name of desired model - in the model-{number} format')
parser.add_argument('--seed', type=str, default=seed_string, help='String to start the text generation - try to avoid characters not used in the original text'  )
args = parser.parse_args()

if args.data_path:
    data_path = args.data_path

if args.length:
    num_predict = args.length

if args.model:
    model_number = args.model

if args.seed:
    seed_string = args.seed

#Load in the text data
_, _, vocabulary, reversed_dictionary, char_to_id = load_data(data_path)

model = load_model(data_path + "model-" + model_number + ".hdf5")

def generate_text(model, input_length, output_length, vocab_size, reversed_dictionary):
    print("Generated Text: ")
    y_char = []
    X = np.random.random_integers(low=0, high=vocab_size - 1, size=(1, input_length))
    if len(seed_string):
        print(seed_string, end="")
        seedIds = [char_to_id[char] for char in seed_string if char in char_to_id]
        if len(seedIds) > input_length:
            X[0] = seedIds[-input_length:]
        else:
            X[0][-len(seedIds):] = seedIds
    for _ in range(output_length):
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