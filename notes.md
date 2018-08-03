# Sci-fi NN
## Get the data:

Project Gutenberg Science Fiction CD

## Combine the data:
Remove other languages, remove the gutenberg transcribers notes and other info

## LSTM: 

https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537

### 6 hrs of training and ...

    all the things of I I Bartleby, I I Bartleby, I I Bartleby, I I Bartleby, I I Bartleby, I I Bartleby,

## Switch to Keras:

http://adventuresinmachinelearning.com/keras-lstm-tutorial/

Epochs are a single pass over the data set - the bigger the data set the longer a single epoch takes. So, with my initial data set of 40MB - a single epoch takes 80 hours on a CPU - so let's try using a few gpus.

## Keras on Google Cloud:
Since I didn't actually have access to a few GPUs - I turned to the cloud. Luckily I still had money left on my Google Cloud free trial. So let's find a guide to using Keras on the cloud.

http://liufuyang.github.io/2017/04/02/just-another-tensorflow-beginner-guide-4.html

issues with getting it to run as a module -> kerasLSTM/
tensorflow version - want 1.8 not 1.0 (which is default)

### 40 hours of training -> ~4 epochs :

    '"white "white <eos> "white...' this is ~9% accurate I guess

So, I'd likely need to train it much much longer ~ 50 epochs but I don't have 500 free gpu hours on Google Cloud nor the equivalent $200 to spare, so we need another option

## Char based Keras:
https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/

http://karpathy.github.io/2015/05/21/rnn-effectiveness/

At initial glance this method is much better for my data - it takes about 1 hr per epoch AND and has accuracy almost immediately above 10%!

## Proof of Concept trained on the writings of Melville:

### 1 epoch:
    the stranger of the sailors of the sailors of the sailors of the stranger of the stranger of the stranger of the stranger of the stranger of

### 26 epochs (~60% Categorical Accuracy):
    THE COMMODORE OF THE STRANGER IN A MAN-OF-WAR.


    The ship was a sort of considerable concern to the sea, the ship's company were seated by

### 50 epochs (~65% Categorical Accuracy):
    The master-at-arms ashore has been seen at the main-mast, and the sailors were all the sailors who had been seen at the main-mast

### 61 has the highest categorical accuracy so far (as of epoch 100) -> 65.41%

    is the Captain of the officers of the Purser's Steward as a common sailor, and the sailors were set forth in the ship's boat, and the same strange strict and solemn of the sea and the same strange strip of a strange sort of state as the same strange strip of state as the ship is almost a sort of strange dog the same thing to be a sort of state of sea-waters

## Begin Training on Sci-fi Data - Target Accuracy ~65%
The training data for the Melvile works above was about 3.5Mb and each epoch took about 50 min. The sci-fi data is considerably bigger - 12.7Mb (which is reduced from the originaly planned 40mb) so each epoch takes about 4 hours.

## 2 Epochs:
    te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te



## Load into tensorflow.js:
