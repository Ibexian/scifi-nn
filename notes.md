# Sci-fi NN

After reading about [how amazing LSTM and Word2Vec are at Natural Language Processing](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) and taking a few online courses on Neural Network basics - I decided I had to try it out for myself. But, with any machine learning / NN modeling project what data to use and where to get it are always some of the biggest questions of the whole project. Luckily for me Project Guteberg exists.
## Get the data:
[Project Gutenberg's](https://www.gutenberg.org/) public domain library of books is an incredible resource for all kinds of books and their [catalogue of Science Fiction](https://www.gutenberg.org/wiki/Science_Fiction_(Bookshelf)) is no exception. They've even collected most of the materials into a downloadable CD, which, since we're looking for lots of Sci-Fi to base the neurnal network on, is exactly what we need.

## Combine the data:
Ok, well not quite 'exactly what we need' - the txt files that you get as part of the collection have a few issues for us to deal with. They're not all in English, they have lots of transcriber's notes and legal liscences attached, and each book is a separate file.
In order to get our data ready for the project these were issues I had to deal with. Since each file said its language near the top of the file - going through and deleting all non-English files was pretty trivial. 

The transcription notes and legal notices on the other hand proved a bit too stubborn for me to automate their removal - since many had different formats, locations, and content - the lack of consistancy made it easier to just remove this info by hand.

For the final bit of data manipulation I wanted to combine the separate works into three files to use in the training, validating, and testing - this was easy enough to manage with a little python file combiner I've creatively called [`fileCombiner.py`](fileCombiner.py).

With our three training files ready to go (you can see the training file [here](input_data/combined.train.txt)) - we could now move on to the actual model creation work.

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

### The Trouble with NaN
When trying to train on the sci-fi with the same params - (seq length - 140, batch length - 20, two lstm layers etc.) Training would get to a certain point (usually 35% accuracy) and the loss would change to `NaN` and the accuracy would plummet.

To avoid needlessly running the training with a `NaN` value for loss Keras has a nice helper function you can bind as a callback to `model.fit` called `TerminateOnNaN` which saves the model and stops training if the loss becomes `NaN`.

But, just stopping my training isn't the goal - so I changed the optimization method (adadelta -> adam -> rmsprop), activation methods (softmax -> relu -> both), the batch size (140 -> 30 -> 50), the step size (20 -> 128), the ammount of data normalization (0 -> 3 -> 1) and in the end found a combination that got passed the 35% accuracy mark (without NaN for the loss). Each data set can be a bit different, so these might require some adjustment for other text data.

It seems the NaN issue is some combination of the learning rate, the batch size, and the momentum (determined by the optimization method) - I can decrease the learning rate or increase the batch size as the epoch increases in order to avoid getting NaN. Momentum trains faster, but can also lead to "hill climbs" where the loss rate increases rather than decreases.

### 1 Epoch (~51% accurate):
    and the strange strange strange straight things that had been a strange strange strange strange strange strange straight through the strange

### 10 Epochs (~62% Categorical Accuracy):
    "Well, there is a strange thing to the stars, and the cadets were all right. The cadets were all right, and the cadets were all right



## Load into tensorflow.js:
