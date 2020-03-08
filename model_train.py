from get_graph import create_graph, create_graph_debug 

import pandas as pd 
import numpy as np
import keras
from sklearn.feature_extraction.text import TfidfVectorizer 
import pickle
import os
from collections import OrderedDict
import networkx as nx

from tqdm import tqdm
from itertools import combinations
import math

# FILE PATH adjusted for colab
def save_as_pickle(filename, data):
    completeName = os.path.join("drive/My Drive/Coding/GCN/data/", filename)
    print(completeName)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)
        
def load_pickle(filename):
    completeName = os.path.join("drive/My Drive/Coding/GCN/data", filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


def load_data(load_from_disc, debug, test_size = None, train_size = None, vocab_size = None):
    """
    load_from_disc: load data from s3 or from own disk; the later ther must be stored priorly, 
    debug: defines if the train and test data filtered, 
    test_size: size of the test data  (Debug = True)
    train_size = size of the train data  (Debug = True)
    vocab_size = size of the vocabolary (load_from_disc = False)
    """
    global idx_train, idx_test

    if load_from_disc:
        docs = load_pickle("data_joined.pkl")
        y_train = load_pickle("y_train.pkl")
        y_test = load_pickle("y_test.plk")

        # Index for semi supervised test data 
        idx_test = [i for i in range(len(y_train), len(y_train) + len(y_test) )]
        idx_train = [i for i in range(len(y_train))]

        if debug:
            # DEBUG!!!
            idx_train = idx_train[:train_size]
            idx_test = idx_test[:test_size] 

            X_train = [docs[i] for i in idx_train]
            X_test = [docs[i] for i in idx_test]
            docs = np.append(X_train, X_test)

            y_train = [y_train[i] for i in [idx_train]]
            y_test = y_test[:test_size]


    else:
        vocab_size = vocab_size
        imdb = keras.datasets.imdb

        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size)

        if debug:
            # DEBUG!!!
            X_train = X_train[:train_size]
            y_train = y_train[:train_size]

            X_test = X_test[:test_size]
            y_test = y_test[:test_size]

        # Decode imdb index to words
        imdb_word_index = imdb.get_word_index()
        imdb_word_index = {k: (v + 3) for k,v in imdb_word_index.items()}
        imdb_word_index["<PAD>"] = 0
        imdb_word_index["<START>"] = 1
        imdb_word_index["<UNK>"] = 2
        imdb_word_index["<UNUSED>"] = 3

        imdb_index_word = {idx: value for value, idx in imdb_word_index.items()}

        def decode_text(txt):
            return ' '.join([imdb_index_word.get(i, '?') for i in txt])


        # Create data for semi-supervised classifier
        data = np.append(X_train, X_test)
        docs = list(map(lambda i: decode_text(i), data )) # decode idx to text


        # Index for semi supervised test data 
        idx_test = [i for i in range(len(y_train), len(y_train) + len(y_test) )]
        idx_train = [i for i in range(len(y_train))]

        # Store results
        save_as_pickle("data_joined.pkl", docs)
        save_as_pickle("y_train.pkl", y_train)
        save_as_pickle("y_test.plk", y_test)
    

    return docs, y_train, y_test


# Fetch Data
docs, y_train, y_test = load_data(load_from_disc = True, debug = True, train_size = 10000, test_size = 10000, vocab_size = 5000)


# Create grapgh first time 
# G, word_word, pmi_ij = create_graph(docs)

# save_as_pickle("text_graph.pkl", G)
# save_as_pickle("word_word.pkl", word_word)
# save_as_pickle("pmi.pkl", pmi_ij)

# Load grapgh from disk
G = load_pickle("text_graph.pkl")
word_word = load_pickle("word_word.pkl")
pmi_ij = load_pickle("pmi.pkl")


# # Adjacancy matrix
# A = nx.to_numpy_matrix(G, weight = "weight")
# A = A + np.eye(G.number_of_nodes()) # diag = 1

# # Degree matrix
# degrees = []
# for d in G.degree(weight = None):
#     if d == 0:
#         degrees.append(0)
#     else:
#         degrees.append(d[1]**(-0.5))
# degrees = np.diag(degrees)

# # A hat
# A_hat = degrees@A@degrees \

# # Feature matrix
# X = np.eye(G.number_of_nodes()) # Features are just identity matrix
# f = X # input of the net


# save_as_pickle("f.pkl", f)
# save_as_pickle("A_hat.pkl", A_hat)

f = load_pickle("f.pkl")
A_hat = load_pickle("A_hat.pkl")


# Build, Train, Test

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected

# Network Parameters
n_input = f.shape[0] # Input each node in the graph as one-hot encoding
n_classes = 2
dropout = 0.75
n_hidden_1 = 330 # Size of first GCN hidden weights
n_hidden_2 = 130
learning_rate = 0.01

# Graph inputs
X = tf.placeholder(tf.float32, [n_input, n_input])
y = tf.placeholder(tf.float32,  [None, n_classes])

keep_prob  = tf.placeholder(tf.float32)
idx_selected = tf.placeholder(tf.int32, [None])

A_hat_tf = tf.convert_to_tensor(A_hat, dtype = tf.float32)
A_hat_tf = tf.Variable(A_hat_tf)


weights = {
    'h1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
}


biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2]))
}


def convLayer(X, A_hat_tf, w, b ):
    X = tf.add(tf.matmul(X, w), b)  # [?,440][440, 330] + [330] = 440x330
    X = tf.matmul(A_hat_tf, X)  # [440, 440][440,330] = [440,330]
    
    return tf.nn.relu(X)


def gcn(X, weights, biases, A_hat, dropout):

    # First convolution layer ?x330
    conv1 = convLayer(X, A_hat_tf, weights['h1'], biases["b1"])
    # Second convolution layer 330x130
    conv2 = convLayer(conv1, A_hat_tf, weights['h2'], biases["b2"])
    # Fully connected layer / Linear layer for logit
    logits = fully_connected(conv2, n_classes, activation_fn = None)
    # Apply Dropout
    #logits = tf.nn.dropout(logits, dropout)
    
    return logits
    

# Build the GCN
pred = gcn(X, weights, biases, A_hat_tf, dropout)

#Filter training document nodes for semi-supervised learning
pred = tf.gather(pred, indices = idx_selected)

# Define optimizer and loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, axis = 1), tf.argmax(y, axis = 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Start trainings session
init  = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    
    sess.run(init)
    
    for e in range(1000):
        
        batch_x = f # one hot for each node (word + docs) in the graph
        batch_y = np.eye(n_classes)[y_train]
        
        sess.run(optimizer, feed_dict = {X: batch_x,
                                         y: batch_y,
                                         keep_prob: dropout,
                                         idx_selected: idx_train})
        
        loss, acc = sess.run([cost, accuracy], feed_dict = {X: batch_x,
                                                            y: batch_y,
                                                            keep_prob: 1.,
                                                            idx_selected: idx_train})
        
        print("Epoch ", e, "Batch size: ", batch_y.shape[0] ,"Batch loss: ", loss, "Training accuracy: ", acc)
    
    # save_path = saver.save(sess, "drive/My Drive/Coding/GCN/data/model.ckpt")

    print("Finish!")

    print("Store model weights and biases")
    weights_dict = {}
    for key, values in weights.items():
        weights_dict[key] = sess.run(values)
    save_as_pickle("weights.pkl", weights_dict)

    biases_dict = {}
    for key, values in biases.items():
        biases_dict[key] = sess.run(values)
    save_as_pickle("biases.pkl", biases_dict)

    # Calculate acc for test docs
    batch_x = f
    batch_y = np.eye(n_classes)[y_test]
    test_acc = sess.run(accuracy, feed_dict = {X: batch_x,
                                               y: batch_y,
                                               keep_prob: 1.,
                                               idx_selected: idx_test})
    print("Testing accuracy: ", test_acc, "Batch size: ", batch_y.shape[0])