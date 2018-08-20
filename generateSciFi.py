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
parser.add_argument('--seed', type=str, default=seed_string, help='String to start the text generation - try to avoid characters not used in the original text')
parser.add_argument('--final', action='store_true', default=False)
args = parser.parse_args()

if args.data_path:
    data_path = args.data_path

if args.length:
    num_predict = args.length

if args.final:
    model_loc = "final_model.hdf5"
else:
    model_loc = "model-" + args.model + ".hdf5"

if args.seed:
    seed_string = args.seed

#Load in the text data
_, _, vocabulary, reversed_dictionary, char_to_id = load_data(data_path)

model = load_model(data_path + model_loc)

def sample(preds, temperature=1.0):
    # Modify the prediction array by temp - to avoid always picking the best prediction
    # Higher temp will result in more varied prediction results
    preds = np.asarray(preds).astype('float64')
    # ignore the log of zero issues
    np.seterr(divide='ignore')
    preds = np.log(preds) / temperature
    # reenable the warn
    np.seterr(divide='warn')
    exp_preds = np.exp(preds)
    # Normalize the probability inputs
    preds = exp_preds / np.sum(exp_preds)
    # Create an array of probabilities
    probabilities = np.random.multinomial(1, preds, 1)
    return np.argmax(probabilities)

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
        predictions = model.predict(X)[:, input_length - 1, :][0]
        ix = sample(predictions, 0.5)
        # Assign that prediction as the last val in X
        X[0][0] = ix
        X = np.roll(X, -1, axis=1)
        # Add Predition to output
        print(reversed_dictionary[ix], end="")
        y_char.append(reversed_dictionary[ix])
    return('').join(y_char)

generate_text(model, num_steps, num_predict, vocabulary, reversed_dictionary)
print("")