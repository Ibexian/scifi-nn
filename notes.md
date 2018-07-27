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

80 hrs to train one epoch... yikes

## Keras on Google Cloud:

http://liufuyang.github.io/2017/04/02/just-another-tensorflow-beginner-guide-4.html

issues with getting it to run as a module -> kerasLSTM/
tensorflow version - want 1.8 not 1.0 (which is default)

### 40 hours of training -> ~4 epochs :

    '"white "white <eos> "white...' this is ~9% accurate I guess

So, I'd likely need to train it much much longer ~ 50 epochs but I don't have 500 free gpu hours on Google Cloud nor the equivalent $200 to spare, so we need another option

## Char based Keras:
https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/

http://karpathy.github.io/2015/05/21/rnn-effectiveness/

~1 hr per epoch AND and accuracy above 10%!

### 1 epoch - trained on Moby Dick as an example:
    tn thore, t shought t hauld heyl fbout t sittle abd see the sater
    cart of the sarld, Inws a ser o have nntaeaving tff toe shieen tnd semu

### 26 epochs (~60% Categorical Accuracy) - trained on Moby Dick:
    THE COMMODORE OF THE STRANGER IN A MAN-OF-WAR.


    The ship was a sort of considerable concern to the sea, the ship's company were seated by

### 50 epochs (~65% Categorical Accuracy) - trained on Moby Dick:
    The master-at-arms ashore has been seen at the main-mast, and the sailors were all the sailors who had been seen at the main-mast

## Load into tensorflow.js: