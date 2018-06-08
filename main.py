import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

with open('test.txt', 'r') as myfile:
    corpus_raw=myfile.read().replace('\n', '').replace('.', ' . ')

# convert to lower case
corpus_raw = corpus_raw.lower()

words = []

for word in corpus_raw.split():
    if word != '.': #because we don't want to treat . as a word
        words.append(word)

words = set(words) #so that all duplicate words are removed

word2int = {}
int2word = {}

vocab_size = len(words) # gives the total number of unique words

for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

#raw sentences is a list of sentences.
raw_sentences = corpus_raw.split('.')

sentences = []

for sentence in raw_sentences:
    sentences.append(sentence.split())

data = []

WINDOW_SIZE = 2

for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :
            if nb_word != word:
                data.append([word, nb_word])

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

y_train = [] #input word
x_train = [] #output word

for data_word in data:
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

print(x_train.shape, y_train.shape)

# making placeholders for x_train and y_train
x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

EMBEDDING_DIM = 5

W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal( [EMBEDDING_DIM]))

hidden_representation = tf.add(tf.matmul(x, W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))

prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))

#Train the model

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

#define the loss function
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))

#define the training step
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)

n_iters = 10000

#train for n_iter iterations
for _ in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    print("loss is : ", sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))
    print(_)

vectors = sess.run(W1 + b1)

# Find the closest words in the vectors
def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2)**2))

def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = 1

    query_vector = vectors[word_index]

    for index, vector in enumerate(vectors):

        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index


# Graph the results
from sklearn.manifold import TSNE
from sklearn import preprocessing

model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
vectors = model.fit_transform(vectors)

normalizer = preprocessing.Normalizer()
vectors = normalizer.fit_transform(vectors, 'l2')

fig, ax = plt.subplots()
print(words)
for word in words:
    print(word, vectors[word2int[word]][1])
    ax.annotate(word, (vectors[word2int[word]][0], vectors[word2int[word]][1]))

plt.show()
#TODO Save Model
#TODO Negative Sampling for W2V
#TODO Hierarchical Softmax

#TODO Eventually switch to gensim (https://radimrehurek.com/gensim/models/word2vec.html)